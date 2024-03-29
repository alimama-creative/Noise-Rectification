from .pipeline_animation import *  # Take from Animatediff: https://github.com/guoyww/AnimateDiff/tree/main
import PIL.Image

def preprocess_image(image, width, height):
    assert isinstance(image, PIL.Image.Image)
    image = np.array(image.resize((width, height))).astype(np.float32) / 255.0
    image = np.expand_dims(image, 0)
    image = image.transpose(0, 3, 1, 2)
    image = 2.0 * image - 1.0
    image = torch.from_numpy(image)
    return image

class NoiseRectificationI2V_Pipeline(AnimationPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        controlnet = None,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, controlnet)
    
    def prepare_latents(self, input_image, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Our method first adds noise to the input image and keep the added noise for latter rectification.
        noise = latents.clone()
        if input_image is not None:
            input_image = preprocess_image(input_image, width, height)
            input_image = input_image.to(device=device, dtype=dtype)

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(input_image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(input_image).latent_dist.sample(generator)
        else:
            init_latents = None

        if init_latents is not None:
            init_latents = rearrange(init_latents, '(b f) c h w -> b c f h w', b = batch_size, f = 1)
            init_latents = init_latents.repeat((1, 1, video_length, 1, 1)) * 0.18215
            noisy_latents = self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0])

        return noisy_latents, noise

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        
        input_image = None,
        noise_rectification_period: Optional[list] = None,
        noise_rectification_weight: Optional[torch.Tensor] = None,
        noise_rectification_weight_start_omega = 1.0,
        noise_rectification_weight_end_omega = 0.5,

        **kwargs,
    ):    
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        noisy_latents, noise = self.prepare_latents(
            input_image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = noisy_latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, 
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals = None,
                    mid_block_additional_residual   = None,
                ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # [The core code of our method.]
                # our method rectifies the predicted noise with the GT noise to realize image-to-video.
                if noise_rectification_period is not None:
                    assert len(noise_rectification_period) == 2
                    if noise_rectification_weight is None:
                        noise_rectification_weight = torch.cat([torch.linspace(noise_rectification_weight_start_omega, noise_rectification_weight_end_omega, video_length//2), 
                                                                torch.linspace(noise_rectification_weight_end_omega, noise_rectification_weight_end_omega, video_length//2)])
                    noise_rectification_weight = noise_rectification_weight.view(1, 1, video_length, 1, 1)
                    noise_rectification_weight = noise_rectification_weight.to(latent_model_input.dtype).to(latent_model_input.device)

                    if i >= len(timesteps) * noise_rectification_period[0] and i < len(timesteps) * noise_rectification_period[1]:
                        delta_frames = noise - noise_pred
                        delta_noise_adjust = noise_rectification_weight * (delta_frames[:,:,[0],:,:].repeat((1, 1, video_length, 1, 1))) + \
                                            (1 - noise_rectification_weight) * delta_frames
                        noise_pred = noise_pred + delta_noise_adjust

                # compute the previous noisy sample x_t -> x_t-1
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, noisy_latents)

        # Post-processing
        video = self.decode_latents(noisy_latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

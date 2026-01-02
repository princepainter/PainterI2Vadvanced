import torch
import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override

class PainterI2VAdvanced(io.ComfyNode):
    """Enhanced Wan2.2 I2V node with post-correction color drift prevention for dual-sampler workflow"""
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterI2VAdvanced",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("motion_amplitude", default=1.3, min=1.0, max=2.0, step=0.05),
                io.Boolean.Input("color_protect", default=True),
                io.Float.Input("correct_strength", default=0.01, min=0.0, max=0.3, step=0.01),
                io.ClipVisionOutput.Input("clip_vision", optional=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="high_positive"),
                io.Conditioning.Output(display_name="high_negative"),
                io.Conditioning.Output(display_name="low_positive"),
                io.Conditioning.Output(display_name="low_negative"),
                io.Latent.Output(display_name="latent"),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.3, color_protect=True, 
                correct_strength=0.05, start_image=None, clip_vision=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        positive_original = positive
        negative_original = negative
        
        if start_image is not None:
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            
            image = torch.ones((length, height, width, start_image.shape[-1]), 
                             device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], 
                             concat_latent_image.shape[-1]), 
                            device=start_image.device, dtype=start_image.dtype)
            mask[:, :, 0] = 0.0
            
            concat_latent_image_original = concat_latent_image.clone()
            
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
                
                post_enhanced = concat_latent_image.clone()
                
                if color_protect and correct_strength > 0:
                    orig_mean = concat_latent_image_original.mean(dim=(2, 3, 4))
                    enhanced_mean = post_enhanced.mean(dim=(2, 3, 4))
                    
                    mean_drift = torch.abs(enhanced_mean - orig_mean) / (torch.abs(orig_mean) + 1e-6)
                    problem_channels = mean_drift > 0.18
                    
                    if problem_channels.any():
                        drift_amount = enhanced_mean - orig_mean
                        correction = drift_amount * problem_channels.float() * correct_strength * 0.03
                        
                        for b in range(batch_size):
                            for c in range(16):
                                if correction[b, c].abs() > 0:
                                    post_enhanced[b, c] = torch.where(
                                        post_enhanced[b, c] > 0,
                                        post_enhanced[b, c] - correction[b, c],
                                        post_enhanced[b, c]
                                    )
                    
                    orig_brightness = concat_latent_image_original.mean()
                    enhanced_brightness = post_enhanced.mean()
                    
                    if enhanced_brightness < orig_brightness * 0.92:
                        brightness_boost = min(orig_brightness / (enhanced_brightness + 1e-6), 1.05)
                        post_enhanced = torch.where(
                            post_enhanced < 0.5,
                            post_enhanced * brightness_boost,
                            post_enhanced
                        )
                    
                    concat_latent_image = torch.clamp(post_enhanced, -6, 6)
            
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            
            positive_original = node_helpers.conditioning_set_values(
                positive_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask}
            )
            negative_original = node_helpers.conditioning_set_values(
                negative_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask}
            )
            
            ref_latent = vae.encode(start_image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"reference_latents": [ref_latent]}, append=True)
            negative_original = node_helpers.conditioning_set_values(negative_original, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision})
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"clip_vision_output": clip_vision})
            negative_original = node_helpers.conditioning_set_values(negative_original, {"clip_vision_output": clip_vision})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, positive_original, negative_original, out_latent)


class PainterI2VAdvancedExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterI2VAdvanced]

async def comfy_entrypoint() -> PainterI2VAdvancedExtension:
    return PainterI2VAdvancedExtension()


NODE_CLASS_MAPPINGS = {
    "PainterI2VAdvanced": PainterI2VAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterI2VAdvanced": "PainterI2VAdvanced",
}

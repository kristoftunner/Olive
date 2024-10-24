# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

image_width = 640
image_height = 352
vae_sample_width = image_width
vae_sample_height= image_height
unet_sample_width = int(image_width / 8)
unet_sample_height = int(image_height / 8)
cross_attention_dim = 768

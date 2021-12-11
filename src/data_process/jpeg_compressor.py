import imgaug.augmenters as iaa


def jpeg_compress(np_image):
    """Preprocess Image with JPEG compression"""
    img_list = [np_image.astype("uint8")]

    # 60 - 75 means, a quality of 40 - 25
    seq_free = iaa.Sequential(
        [
            iaa.JpegCompression(compression=(60, 75)),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
        ]
    )

    return seq_free(images=img_list)[0].astype("float32")

from torchvision import transforms 

def custom_transform_with_selective_augs(is_train, args):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train and args.aug == "yes":
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=30,
                translate=(0.1, 0.1),
                scale=(0.95, 1.15),
                shear=0
            ),
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224))
        ]

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return transforms.Compose(transform_list)
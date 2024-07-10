import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg

def calculate_activation_statistics(images, model, batch_size=1, dims=1000):
    model.eval()
    n_batches = len(images) // batch_size
    act = np.empty((len(images), dims))

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((299, 299)),  # Resize images to the input size of InceptionV3
    ])

    for i in range(n_batches):
        batch = images[i * batch_size: (i + 1) * batch_size]
        # print(batch.shape)

        # for img in batch:
        #     print(transform(img.cpu().numpy()).shape)
        #     exit(0)
        batch = torch.stack([transform(img.cpu().numpy()).reshape(1, 3, 96, 96) for img in batch])
        batch = batch.to('cuda') if torch.cuda.is_available() else batch
        # print(act.shape)
        # exit(0)
        with torch.no_grad():
            batch = batch.squeeze(0)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)  # Upsample to 299x299
            # print(batch.shape)
            # exit(0)
            pred = model(batch)[0]
            # print(pred.shape)
            # exit(0)
            # If using InceptionV3, the output is a tuple where the first element is the logits
            act[i * batch_size: i * batch_size + len(batch)] = pred.cpu().numpy()

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "The means of the two distributions must have the same shape"
    assert sigma1.shape == sigma2.shape, "The covariance matrices of the two distributions must have the same shape"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            #raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class FID():
    def __init__(self, model):
        self.model = model

    def calculate_fid(self, images1, images2, batch_size=1):
        mu1, sigma1 = calculate_activation_statistics(images1, self.model, batch_size=batch_size)
        mu2, sigma2 = calculate_activation_statistics(images2, self.model, batch_size=batch_size)
        fid_value = frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value




# import torch
# import torchvision.transforms as transforms
# from torchvision.models import inception_v3
# import numpy as np
# from scipy.linalg import sqrtm
# from PIL import Image
# import torch.nn.functional as F
# import torch.nn as nn

# # Adjusted Inception v3 model to support 96x96 images
# class InceptionV3_96(nn.Module):
#     def __init__(self, pretrained=True):
#         super(InceptionV3_96, self).__init__()
#         self.inception_v3 = inception_v3(pretrained=pretrained, transform_input=False)
#         self.inception_v3.fc = nn.Identity()
#         self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((3, 3))  # Adjust pooling layer
#         self.conv_layer = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=204, stride=3, padding=0)

#     def forward(self, x):
#         x = x.reshape(3, 96, 96).cuda()

#         x = x.unsqueeze(0)
#         x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)  # Upsample to 299x299
#         # x = self.conv_layer(x)
#         x = self.inception_v3(x.cuda())
#         # print(x.shape)
#         # exit(0)
#         # x = self.adaptive_avg_pool(x)
#         # x = x.view(x.size(0), -1)
#         return x.reshape((2048))
    

# # Function to compute inception features
# def compute_inception_features(images, batch_size=1):
#     # Load Inception v3 model pretrained on ImageNet
#     inception = InceptionV3_96().cuda().eval()
#     # inception = inception_v3(pretrained=True, transform_input=False).cuda().eval()
#     # inception.fc = torch.nn.Identity()  # Remove classification layer

#     # Preprocess images
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Resize(299),
#         # transforms.CenterCrop(299)
#     ])

#     # Compute inception features
#     features = []

#     with torch.no_grad():
#         for i in range(len(images)):
#             # print(images[i].shape)
#             # exit(0)
#             # print(transform(images[i].cpu().numpy()).shape)
#             # exit(0)
#             batch_features = inception(transform(images[i].cpu().numpy()))
#             features.append(batch_features.cpu().numpy())
#     features = np.concatenate(features, axis=0)
#     return features

#     # with torch.no_grad():
#     #     num_batches = (len(images) + batch_size - 1) // batch_size
#     #     for i in range(num_batches):
#     #         batch = images[i * batch_size: (i + 1) * batch_size].cuda()
#     #         batch_features = inception(transform(batch.cpu().numpy()))
#     #         features.append(batch_features.cpu().numpy())
#     # features = np.concatenate(features, axis=0)
#     # return features

# # Function to compute FID
# def calculate_fid(images1, images2, batch_size=1):
#     features1 = compute_inception_features(images1, batch_size)
#     features2 = compute_inception_features(images2, batch_size)
    
#     print(features1.shape)
#     # Compute mean and covariance
#     mean1, cov1 = np.mean(features1, axis=0), np.cov(features1, rowvar=True)
#     mean2, cov2 = np.mean(features2, axis=0), np.cov(features2, rowvar=True)

#     print(cov1)
#     print(cov2.shape)
#     # exit(0)
#     # Compute Frechet distance
#     diff = mean1 - mean2
#     cov_sqrt = sqrtm(cov1.dot(cov2))
#     if np.iscomplexobj(cov_sqrt):
#         cov_sqrt = cov_sqrt.real
#     fid = np.dot(diff, diff) + np.trace(cov1 + cov2 - 2 * cov_sqrt)
#     return fid

# # Example usage
# # images1 and images2 should be PyTorch tensors of shape (N, 3, 96, 96)
# # images1 = torch.randn(100, 3, 96, 96)  # Example random images
# # images2 = torch.randn(100, 3, 96, 96)  # Example random images
# # fid_score = calculate_fid(images1, images2)
# # print("FID Score:", fid_score)

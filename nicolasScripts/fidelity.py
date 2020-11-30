from torch_fidelity import calculate_metrics

#runs the inception pretrained classifier and looks how easily it can distinguish different stuff in the generated images
#input1 = first path to sample of images
#input2 = second path to sample of images
#cuda = gpu usage
#isc = inception score
#kid = kernel inception distance
#fid = frechet inception distance
pretrained_metrics = calculate_metrics("./results/facades_label2photo_pretrained/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
trained_metrics = calculate_metrics("./results/perceptual/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)

print("pretrained metrics", pretrained_metrics)
print("trained metrics for perceptual", trained_metrics)


from torch_fidelity import calculate_metrics

#runs the inception pretrained classifier and looks how easily it can distinguish different stuff in the generated images
#input1 = first path to sample of images
#input2 = second path to sample of images
#cuda = gpu usage
#isc = inception score
#kid = kernel inception distance
#fid = frechet inception distance
adv_og_metrics = calculate_metrics("./results/adv_og/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
adv_hinge_metrics = calculate_metrics("./results/adv_hinge/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
adv_w_metrics = calculate_metrics("./results/adv_w/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)
adv_ls_metrics = calculate_metrics("./results/adv_ls/test_latest/images/", "./datasets/facades/test/", cuda=False, isc=True, fid=True, kid=False, verbose=True)

print("adv_og_metrics", adv_og_metrics)
print("adv_hinge_metrics", adv_hinge_metrics)
print("adv_w_metrics", adv_w_metrics)
print("adv_ls_metrics", adv_ls_metrics)

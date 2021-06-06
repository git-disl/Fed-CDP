

import numpy as np
import csv




##### if the sigma is heterogeneous need to load the sigma_list from the federated learning training file

#sigma_list= 6*np.ones(101)

# #####base composition, fixed sigma, random sampling
# # cur_sigma=1
# # eps=np.sqrt(2*np.log(125000))/cur_sigma
# # print (eps)
# # eps=np.log(1+0.1*(np.exp(eps)-1))
# # print (eps)
# eps_sum=0
# #sigma_list=0.5*np.ones(100)
# for k in range(1,102):
#     cur_sigma=sigma_list[k-1]
#     eps=np.sqrt(2*np.log(125000))/float(cur_sigma)
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     eps_sum = eps_sum +eps
#     with open('basecompfix_sampling.csv', mode='a') as basecompfix_sampling_file:
#         writer_basecompfix_sampling = csv.writer(basecompfix_sampling_file, delimiter=',')
#         writer_basecompfix_sampling.writerow([k, eps_sum])
# print (eps_sum)

# # # base composition, fixed sigma, random shuffling
# # eps_sum=0
# # for k in range(1,101):
# #     cur_sigma=sigma_list[k-1]
# #     eps=np.sqrt(2*np.log(125000))/float(cur_sigma)
# #     eps_sum = eps_sum +eps
# #     with open('basecompfix_shuffling.csv', mode='a') as basecompfix_shuffling_file:
# #         writer_basecompfix_shuffling = csv.writer(basecompfix_shuffling_file, delimiter=',')
# #         writer_basecompfix_shuffling.writerow([k, eps_sum])
#
#
#
# # #advanced composition, fixed sigma, random shuflling
# # eps_list = []
# # for k in range(1,101):
# #     print (k)
# #     cur_sigma = sigma_list[k - 1]
# #     eps = np.sqrt(2 * np.log(125000)) / float(cur_sigma)
# #     if k==1:
# #         eps_list.append(eps)
# #     else:
# #         eps_list=np.append(eps_list,eps)
# #     with open('advcompfix_shuffling.csv', mode='a') as advcompfix_shuffling_file:
# #         writer_advcompfix_shuffling = csv.writer(advcompfix_shuffling_file, delimiter=',')
# #         writer_advcompfix_shuffling.writerow([k, np.sqrt(np.sum([2*(item**2)*np.log(1e5) for item in eps_list]))+np.sum([(np.exp(item)-1)*item/(np.exp(item)+1) for item in eps_list])])
# #
# # advanced composition, fixed sigma, random sampling
# eps_list = []
# #sigma_list=0.5*np.ones(100)
# for k in range(1, 102):
#     print (k)
#     cur_sigma = sigma_list[k - 1]
#     eps = np.sqrt(2 * np.log(125000)) / float(cur_sigma)
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     if k == 1:
#         eps_list.append(eps)
#     else:
#         eps_list = np.append(eps_list, eps)
#     with open('advcompfix_sampling.csv', mode='a') as advcompfix_sampling_file:
#         writer_advcompfix_sampling = csv.writer(advcompfix_sampling_file, delimiter=',')
#         writer_advcompfix_sampling.writerow([k, np.sqrt(
#             np.sum([2 * (item ** 2) * np.log(1e5) for item in eps_list])) + np.sum(
#             [(np.exp(item) - 1) * item / (np.exp(item) + 1) for item in eps_list])])

#
#
# # #optimal composition, fixed sigma, random sampling
# eps_list = []
# #sigma_list=0.5*np.ones(100)
#
# for k in range(1,102):
#     print (k)
#     cur_sigma = sigma_list[k - 1]
#     eps = np.sqrt(2 * np.log(125000)) / float(cur_sigma)
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     if k==1:
#         sum1 = eps**2
#         sum3 = (np.exp(eps)-1)*eps/(np.exp(eps)+1)
#         eps_list.append(eps)
#     else:
#         sum1 = sum1 + eps**2
#         sum3 = sum3 +(np.exp(eps)-1)*eps/(np.exp(eps)+1)
#         eps_list = np.append(eps_list, eps)
#     with open('optcompfix_sampling.csv', mode='a') as optcompfix_sampling_file:
#         writer_optcompfix_sampling = csv.writer(optcompfix_sampling_file, delimiter=',')
#         writer_optcompfix_sampling.writerow([k, np.sqrt(np.sum([2*(item**2)*np.log(2.71828 + np.sqrt(sum1)*1e5) for item in eps_list])) +sum3])
# #
#
#
# # #optimal composition, fixed sigma, random shuffling
# # eps_list = []
# #
# # for k in range(1,101):
# #     print (k)
# #     cur_sigma = sigma_list[k - 1]
# #     eps = np.sqrt(2 * np.log(125000)) / float(cur_sigma)
# #     if k == 1:
# #         sum1 = eps ** 2
# #         sum3 = (np.exp(eps)-1)*eps/(np.exp(eps)+1)
# #         eps_list.append(eps)
# #     else:
# #         sum1 = sum1 + eps ** 2
# #         sum3 = sum3 +(np.exp(eps)-1)*eps/(np.exp(eps)+1)
# #         eps_list = np.append(eps_list, eps)
# #     with open('optcompfix_shuffling.csv', mode='a') as optcompfix_shuffling_file:
# #         writer_optcompfix_shuffling = csv.writer(optcompfix_shuffling_file, delimiter=',')
# #         writer_optcompfix_shuffling.writerow([k, np.sqrt(np.sum([2*(item**2)*np.log(2.71828 + np.sqrt(sum1)*1e5) for item in eps_list])) +sum3])
#
#
#
#
# # #zcdp, fixed sigma, random shuffling
# # for k in range(1,101):
# #     cur_sigma = sigma_list[k - 1]
# #     if k == 1:
# #         p = 1 / (2 * float(cur_sigma) * float(cur_sigma))
# #     else:
# #         p = p + 1 / (2 * float(cur_sigma) * float(cur_sigma))
# #
# #     with open('zcdpshufflingfix.csv', mode='a') as zcdpshufflingfix_file:
# #         writer_zcdpshufflingfix = csv.writer(zcdpshufflingfix_file, delimiter=',')
# #         writer_zcdpshufflingfix.writerow([k, p+2*np.sqrt(p*11.5129)])
#
#
#
# #zcdp, fixed sigma, random samping
# #sigma_list=0.5*np.ones(100)
# q=0.1
# for k in range(1,102):
#     cur_sigma = sigma_list[k - 1]
#     if k == 1:
#         p = q*q / (float(cur_sigma) * float(cur_sigma))
#     else:
#         p = p + q*q / (float(cur_sigma) * float(cur_sigma))
#
#     with open('zcdpsamplingfix.csv', mode='a') as zcdpsamplingfix_file:
#         writer_zcdpsampingfix = csv.writer(zcdpsamplingfix_file, delimiter=',')
#         writer_zcdpsampingfix.writerow([k, p+2*np.sqrt(p*11.5129)])
# # #
# # #
#moments accountant, fixed sigma, random sampling
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_heterogenous_rdp,get_privacy_spent

eps_list = np.exp(1.664*np.arange(0,101,1)/101-1.1809)
sigma_list = np.sqrt(2*np.log(125000))/eps_list

#print (sigma_list)

#sigma_list=0.5*np.ones(100)
sampling_probabilities = 0.1*np.ones(101)
noise_multipliers = sigma_list[:101]
steps_list = np.ones(101)

orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])


_,rdp_list = compute_heterogenous_rdp(sampling_probabilities,noise_multipliers, steps_list, orders)

for k in range(len(rdp_list)):
    #print (rdp_list[k])
    eps, _, opt_order = get_privacy_spent(orders, rdp_list[k], target_delta=1e-5)
    #print (eps)

    # if (eps>0.147):
    #     print (k)
    #if (eps>0.303):
    #    print (k)
    #if (eps > 0.668):
    #    print(k)
    if (eps > 0.854):
         print(k)
    with open('momentsfix.csv', mode='a') as momentsfix_file:
        writer_momentsfix = csv.writer(momentsfix_file, delimiter=',')
        writer_momentsfix.writerow([k, eps])


# ########################################################## heterogeneous and decay, sigma decay
#
#
# sigma_decay =  -np.arange(0,5000,1)*6/5000+10
# one_list = 4*np.ones(5000)
#
# sigma_list = np.concatenate([sigma_decay,one_list])
# #
# ####base composition, fixed sigma, random sampling
# cur_sigma=sigma_list[5000]
# eps=np.sqrt(2*np.log(125000))/cur_sigma
# print (eps)
# # eps=np.log(1+0.1*(np.exp(eps)-1))
# # print (eps)
# eps_sum=0
# for k in range(1,101):
#     cur_sigma=sigma_list[k-1]
#     eps=np.sqrt(2*np.log(125000))/float(cur_sigma)
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     eps_sum = eps_sum +eps
#     if eps_sum>123.35:
#         print (k)
#         exit()
#     with open('basecompfix_sampling.csv', mode='a') as basecompfix_sampling_file:
#         writer_basecompfix_sampling = csv.writer(basecompfix_sampling_file, delimiter=',')
#         writer_basecompfix_sampling.writerow([k, eps_sum])
#
#
#
# # # # advanced composition, fixed sigma, random sampling
# # eps_list = []
# # for k in range(1, 101):
# #     print (k)
# #     cur_sigma = sigma_list[k - 1]
# #     eps = np.sqrt(2 * np.log(125000)) / float(cur_sigma)
# #     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
# #     if k == 1:
# #         eps_list.append(eps)
# #     else:
# #         eps_list = np.append(eps_list, eps)
# #     eps_sum = np.sqrt(
# #             np.sum([2 * (item ** 2) * np.log(1e5) for item in eps_list])) + np.sum(
# #             [(np.exp(item) - 1) * item / (np.exp(item) + 1) for item in eps_list])
# #     if eps_sum > 7.45:
# #         #print (k)
# #         exit()
# #     with open('advcompfix_sampling.csv', mode='a') as advcompfix_sampling_file:
# #         writer_advcompfix_sampling = csv.writer(advcompfix_sampling_file, delimiter=',')
# #         writer_advcompfix_sampling.writerow([k, np.sqrt(
# #             np.sum([2 * (item ** 2) * np.log(1e5) for item in eps_list])) + np.sum(
# #             [(np.exp(item) - 1) * item / (np.exp(item) + 1) for item in eps_list])])



# #optimal composition, fixed sigma, random sampling
# eps_list = []
# for k in range(1,101):
#     print (k)
#     cur_sigma = sigma_list[k - 1]
#     eps = np.sqrt(2 * np.log(125000)) / float(cur_sigma)
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     if k==1:
#         sum1 = eps**2
#         sum3 = (np.exp(eps)-1)*eps/(np.exp(eps)+1)
#         eps_list.append(eps)
#     else:
#         sum1 = sum1 + eps**2
#         sum3 = sum3 +(np.exp(eps)-1)*eps/(np.exp(eps)+1)
#         eps_list = np.append(eps_list, eps)
#     eps_sum = np.sqrt(np.sum([2*(item**2)*np.log(2.71828 + np.sqrt(sum1)*1e5) for item in eps_list])) +sum3
#     if eps_sum > 6.74:
#         print (k)
#         exit()
#     with open('optcompfix_sampling.csv', mode='a') as optcompfix_sampling_file:
#         writer_optcompfix_sampling = csv.writer(optcompfix_sampling_file, delimiter=',')
#         writer_optcompfix_sampling.writerow([k, np.sqrt(np.sum([2*(item**2)*np.log(2.71828 + np.sqrt(sum1)*1e5) for item in eps_list])) +sum3])




# # #zcdp, fixed sigma, random samping
# q=0.1
# for k in range(1,101):
#     cur_sigma = sigma_list[k - 1]
#     if k == 1:
#         p = q*q / (float(cur_sigma) * float(cur_sigma))
#     else:
#         p = p + q*q / (float(cur_sigma) * float(cur_sigma))
#     eps_sum =  p+2*np.sqrt(p*11.5129)
#     if eps_sum > 1.1588:
#         print(k)
#         exit()
#     with open('zcdpsamplingfix.csv', mode='a') as zcdpsamplingfix_file:
#         writer_zcdpsampingfix = csv.writer(zcdpsamplingfix_file, delimiter=',')
#         writer_zcdpsampingfix.writerow([k, p+2*np.sqrt(p*11.5129)])



# #moments accountant, hetero sigma, random sampling
# from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_heterogenous_rdp,get_privacy_spent
#
# sampling_probabilities = 0.1*np.ones(100)
# noise_multipliers = sigma_list[:100]
# steps_list = np.ones(100)
#
# orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
#             list(range(5, 64)) + [128, 256, 512])
#
#
# _,rdp_list = compute_heterogenous_rdp(sampling_probabilities,noise_multipliers, steps_list, orders)
#
# for k in range(len(rdp_list)):
#     #print (rdp_list[k])
#     eps, _, opt_order = get_privacy_spent(orders, rdp_list[k], target_delta=1e-5)
#     #print (eps)
#
#     if eps > 0.8227:
#         print(k)
#         exit()
#
#     with open('momentsfix.csv', mode='a') as momentsfix_file:
#         writer_momentsfix = csv.writer(momentsfix_file, delimiter=',')
#         writer_momentsfix.writerow([k, eps])





########################################################## heterogeneous and decay, eps decay


#eps_list = np.arange(0,100,1)/100+0.3074675437675648  # linear decay
#eps_list = 0.1*(np.arange(0,100,1)//10)+0.3574675437675648 # staircase decay
#eps_list = np.exp(1.664*np.arange(0,100,1)/100-1.1809)
#sigma_list = np.sqrt(2*np.log(125000))/eps_list


#eps_list = np.log(1 + 0.1 * (np.exp(eps_list) - 1))


#
#np.save('eps_stairdecay_sigma',sigma_list)
#np.savetxt("eps_lineardecay.csv", sigma_list, delimiter=",")
#np.savetxt("eps_lineardecay_before_amp_perstep.csv", eps_list, delimiter=",")
#np.savetxt("eps_staircasedecay_before_amp_perstep.csv", eps_list, delimiter=",")
#np.savetxt("eps_expdecay_before_amp_perstep.csv", eps_list, delimiter=",")
#np.savetxt("eps_lineardecay_after_amp_perstep.csv", eps_list, delimiter=",")
#np.savetxt("eps_staircasedecay_after_amp_perstep.csv", eps_list, delimiter=",")
#np.savetxt("eps_expdecay_after_amp_perstep.csv", eps_list, delimiter=",")
#np.savetxt("eps_staircasedecay.csv", sigma_list, delimiter=",")
# np.savetxt("eps_expdecay.csv", sigma_list, delimiter=",")

#
#
# ###base composition, fixed sigma, random sampling
# eps_sum=0
# # eps=np.sqrt(2*np.log(125000))/6
# # print (eps)
# for k in range(1,101):
#     eps = eps_list[k-1]
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     eps_sum = eps_sum +eps
#     if eps_sum>123.35:
#         print (k)
#         exit()
#     with open('basecompfix_sampling.csv', mode='a') as basecompfix_sampling_file:
#         writer_basecompfix_sampling = csv.writer(basecompfix_sampling_file, delimiter=',')
#         writer_basecompfix_sampling.writerow([k, eps_sum])



# # advanced composition, fixed sigma, random sampling
# amp_eps_list=[]
# for k in range(1, 101):
#     print (k)
#     eps = eps_list[k - 1]
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     if k == 1:
#         amp_eps_list.append(eps)
#     else:
#         amp_eps_list = np.append(amp_eps_list, eps)
#     eps_sum = np.sqrt(
#             np.sum([2 * (item ** 2) * np.log(1e5) for item in amp_eps_list])) + np.sum(
#             [(np.exp(item) - 1) * item / (np.exp(item) + 1) for item in amp_eps_list])
#     if eps_sum > 7.45:
#         #print (k)
#         exit()
#     with open('advcompfix_sampling.csv', mode='a') as advcompfix_sampling_file:
#         writer_advcompfix_sampling = csv.writer(advcompfix_sampling_file, delimiter=',')
#         writer_advcompfix_sampling.writerow([k, np.sqrt(
#             np.sum([2 * (item ** 2) * np.log(1e5) for item in amp_eps_list])) + np.sum(
#             [(np.exp(item) - 1) * item / (np.exp(item) + 1) for item in amp_eps_list])])



# #optimal composition, fixed sigma, random sampling
# amp_eps_list = []
# for k in range(1,101):
#     print (k)
#     eps = eps_list[k - 1]
#     eps = np.log(1 + 0.1 * (np.exp(eps) - 1))
#     if k==1:
#         sum1 = eps**2
#         sum3 = (np.exp(eps)-1)*eps/(np.exp(eps)+1)
#         amp_eps_list.append(eps)
#     else:
#         sum1 = sum1 + eps**2
#         sum3 = sum3 +(np.exp(eps)-1)*eps/(np.exp(eps)+1)
#         amp_eps_list = np.append(amp_eps_list, eps)
#     eps_sum = np.sqrt(np.sum([2*(item**2)*np.log(2.71828 + np.sqrt(sum1)*1e5) for item in amp_eps_list])) +sum3
#     if eps_sum > 6.74:
#         print (k)
#         exit()
#     with open('optcompfix_sampling.csv', mode='a') as optcompfix_sampling_file:
#         writer_optcompfix_sampling = csv.writer(optcompfix_sampling_file, delimiter=',')
#         writer_optcompfix_sampling.writerow([k, np.sqrt(np.sum([2*(item**2)*np.log(2.71828 + np.sqrt(sum1)*1e5) for item in amp_eps_list])) +sum3])




# # #zcdp, fixed sigma, random samping
# q=0.1
# for k in range(1,101):
#     cur_sigma = sigma_list[k - 1]
#     if k == 1:
#         p = q*q / (float(cur_sigma) * float(cur_sigma))
#     else:
#         p = p + q*q / (float(cur_sigma) * float(cur_sigma))
#     eps_sum =  p+2*np.sqrt(p*11.5129)
#     if eps_sum > 1.1588:
#         print(k)
#         exit()
#     with open('zcdpsamplingfix.csv', mode='a') as zcdpsamplingfix_file:
#         writer_zcdpsampingfix = csv.writer(zcdpsamplingfix_file, delimiter=',')
#         writer_zcdpsampingfix.writerow([k, p+2*np.sqrt(p*11.5129)])



# #moments accountant, hetero sigma, random sampling
# from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_heterogenous_rdp,get_privacy_spent
#
# sampling_probabilities = 0.1*np.ones(100)
# noise_multipliers = sigma_list[:100]
# steps_list = np.ones(100)
#
# orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
#             list(range(5, 64)) + [128, 256, 512])
#
#
# _,rdp_list = compute_heterogenous_rdp(sampling_probabilities,noise_multipliers, steps_list, orders)
#
# for k in range(len(rdp_list)):
#     #print (rdp_list[k])
#     eps, _, opt_order = get_privacy_spent(orders, rdp_list[k], target_delta=1e-5)
#     #print (eps)
#
#     if eps > 0.8227:
#         print(k)
#         exit()
#
#     with open('momentsfix.csv', mode='a') as momentsfix_file:
#         writer_momentsfix = csv.writer(momentsfix_file, delimiter=',')
#         writer_momentsfix.writerow([k, eps])

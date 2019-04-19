'''
true_count = 0
for index,point in enumerate(data):
    if np.argmax(r.T[index]) == point[2]:
        true_count+=1
    else:
        print("Error predicting: ",r.T[index]," Label: ",point[2])
print("Percentage: ", true_count*100.0/N)
'''
'''
def calculate_max_likelihood(data,weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= (1/2.0) * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll
'''


'''

correct = 0
for i,p_label in enumerate(predicted_labels):
    if p_label == test_data[i][2]:
        correct+=1
print("Accuracy: ", correct*100.0/len(test_data))
'''

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

'''
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(data,mus,sigmas,r,labels,label=True, ax=None):
    ax = ax or plt.gca()
    labels = predict(data,mus,sigmas,2)
    if label:
        ax.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(data[:, 0], data[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / r.max()
    for pos, covar, w in zip(mus, sigmas, r):
        draw_ellipse(pos, covar)

plot_gmm(data[:,:2],mus,sigmas,r,data[:,2])
plt.show()
'''

'''
def euclidian_distance(train_data,point):
    point = np.array(point)
    differences = train_data - np.ones([len(train_data),point.shape[1]]) * point
    distances = np.sqrt(np.power(differences,2))
    return np.sort(distances)
def predict(train_data,point,k):
    pass
print( euclidian_distance( training_data[:,:2] , [1,2]))
'''

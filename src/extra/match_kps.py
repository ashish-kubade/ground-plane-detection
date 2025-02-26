import immatch
import yaml
from immatch.utils import plot_matches
import sys


# Initialize model
with open('configs/patch2pix.yml', 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)['example']
model = immatch.__dict__[args['class']](args)
matcher = lambda im1, im2: model.match_pairs(im1, im2)

# Specify the image pair
im1 = sys.argv[1]
im2 = sys.argv[2]

# Match and visualize
matches, _, _, _ = matcher(im1, im2)    
plot_matches(im1, im2, matches, radius=2, lines=True)
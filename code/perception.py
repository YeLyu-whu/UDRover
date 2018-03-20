import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


def getRock(img):
    mask = np.zeros_like(img[:,:,0])
    idx = (img[:,:,0]>110)&(img[:,:,0]<230)&(img[:,:,1]>90)&(img[:,:,1]<210)&(img[:,:,2]>=0)&(img[:,:,2]<50)
    mask[idx]=1
    return mask

def getObstacle(img):
    mask = np.zeros_like(img[:,:,0])
    idx = (img[:,:,0]<95)&(img[:,:,1]<95)&(img[:,:,2]<95)
    mask[idx]=1
    return mask

def getTerrain(img):
    mask = np.zeros_like(img[:,:,0])
    idx = (img[:,:,0]>150)&(img[:,:,1]>150)&(img[:,:,2]>150)
    mask[idx]=1
    return mask

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    scale=10
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    world_size = Rover.worldmap.shape[0]
    
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    mask = perspect_transform(np.ones_like(img[:,:,0]), source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    mask_t = getTerrain(warped)*mask
    #mask_o = getObstacle(warped)*mask
    mask_o = (1.0-mask_t)*mask
    mask_t[:mask_t.shape[0]//2,:]=0
    mask_r = getRock(warped)*mask
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = mask_o
    Rover.vision_image[:,:,1] = mask_r
    Rover.vision_image[:,:,2] = mask_t
    # 5) Convert map image pixel values to rover-centric coords
    xpix_t, ypix_t = rover_coords(mask_t)
    xpix_r, ypix_r = rover_coords(mask_r)
    xpix_o, ypix_o = rover_coords(mask_o)
    # 6) Convert rover-centric pixel values to world coordinates
    xpix_t_w,ypix_t_w = pix_to_world(xpix_t, ypix_t, xpos, ypos, yaw, world_size, scale)
    xpix_o_w,ypix_o_w = pix_to_world(xpix_o, ypix_o, xpos, ypos, yaw, world_size, scale)
    xpix_r_w,ypix_r_w = pix_to_world(xpix_r, ypix_r, xpos, ypos, yaw, world_size, scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    if Rover.pitch<3 and Rover.roll<3:
        Rover.worldmap[ypix_t_w, xpix_t_w, 2] = 255
        Rover.worldmap[Rover.worldmap[:,:,2]>0,0] = 0
    Rover.worldmap[ypix_o_w, xpix_o_w, 0] = 255
    Rover.worldmap[ypix_r_w, xpix_r_w, 1] = 255
    
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    ratio = 70/128.0
    mask_t[:,np.int(mask_t.shape[1]*ratio):]=0
    mask_r[:,np.int(mask_t.shape[1]*ratio):]=0
    xpix_r_, ypix_r_ = rover_coords(mask_r)
    Rover.nav_dists = None
    Rover.nav_angles = None
    if len(xpix_r_)>0:
      nav_dists,nav_angles = to_polar_coords(xpix_r_, ypix_r_)
      nav_angles = nav_angles[(nav_dists==np.min(nav_dists))]
      nav_dists = nav_dists[(nav_dists==np.min(nav_dists))]
      nav_angles = nav_angles[(nav_dists<120)]
      nav_dists = nav_dists[(nav_dists<120)]
      #if Rover.nav_dists[0]>15:
       # Rover.nav_angles = np.tile(Rover.nav_angles,Rover.stop_forward)
        #Rover.nav_dists = np.tile(Rover.nav_dists,Rover.stop_forward)
      if nav_angles is not None and len(nav_angles)>0:
        print('Rock! Rock! Rock!')
        #print(len(Rover.nav_dists))
        Rover.nav_dists = nav_dists
        Rover.nav_angles = nav_angles
        if not Rover.picking_up:
          Rover.mode = 'wait'
          global in_wait_state
          in_wait_state = True
    if Rover.nav_dists is None:
      xpix_t_, ypix_t_ = rover_coords(mask_t)
      Rover.nav_dists,Rover.nav_angles = to_polar_coords(xpix_t_, ypix_t_)
      Rover.nav_angles = Rover.nav_angles[(Rover.nav_dists<120)]
      Rover.nav_dists = Rover.nav_dists[(Rover.nav_dists<120)]

    
    return Rover
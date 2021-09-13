#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import math
import time
# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="meanwhile.bag")
# Parse the command line arguments to an object
args = parser.parse_args()

top_border1 = 128
top_border2 = 131

down_border1 = 374
down_border2 = 376


bed_width = 500
"""
def get_bed_border(ar_depth_frame):
    for i in range(0,240,10):
        if (ar_depth_frame.get_distance(0,i)<2.4 and ar_depth_frame.get_distance(500, i)>1):
            a = i
            break

    for i in range(0,240,10):
        if (ar_depth_frame.get_distance(500,i)<2.4 and ar_depth_frame.get_distance(500, i)>1):
            b = i
            break

    for i in range(240,480,10):
        if (ar_depth_frame.get_distance(0,i)>2.4):
            c = i
            break
    for i in range(240,480,10):
        if (ar_depth_frame.get_distance(500,i)>2.4):
            d = i
            break
    return a,b,c,d    
"""
def upper_border_detected(upper_bed_border, ar_depth_frame):

    a = -1
    b = -1
    for i in range(0,600,10):
        if(ar_depth_frame.get_distance(i,upper_bed_border)<1.61 and ar_depth_frame.get_distance(i,upper_bed_border)>1):
            a = i
            break

    for i in range(600,0,-10):
        if(ar_depth_frame.get_distance(i,upper_bed_border)<1.61 and ar_depth_frame.get_distance(i,upper_bed_border)>1):
            b = i
            break

    over_upper_border = False
    if(a>0 and b>0):
        over_upper_border = True
    return a, b, over_upper_border
        
def lower_border_detected(lower_bed_border, ar_depth_frame):
    c = -1
    d = -1  
    for i in range(0,600,10):
        if(ar_depth_frame.get_distance(i,lower_bed_border)<1.61 and ar_depth_frame.get_distance(i,lower_bed_border)>0.5):
            c = i
            break

    for i in range(600,0,-10):
        if(ar_depth_frame.get_distance(i,lower_bed_border)<1.61 and ar_depth_frame.get_distance(i,lower_bed_border)>0.5):
            d = i
            break
    over_lower_border = False
    if(c>0 and d>0):
        over_lower_border = True
        
    return c,d,over_lower_border


def vertical_detected_left(upper_bed_border, lower_bed_border, ar_depth_frame):
        row_min = row_max = col_min = col_max = 0
        
        #depth_01 = ar_depth_frame[upper_bed_border+10:lower_bed_border-10,0:500]
        depth_01 = ar_depth_frame[upper_bed_border+10: lower_bed_border-10,0:bed_width//2]
        depth_bd = np.where((depth_01<1.2))
        up = False
        if(len(depth_bd[0])!=0):
            up = True
            row_min = np.min(depth_bd[0])+upper_bed_border+10
            row_max = np.max(depth_bd[0])+upper_bed_border+10
            col_min = np.min(depth_bd[1])
            col_max = np.max(depth_bd[1])
        return row_min,row_max,col_min,col_max,up


def vertical_detected_right(upper_bed_border, lower_bed_border, ar_depth_frame):
        row_min = row_max = col_min = col_max = 0
        
        #depth_01 = ar_depth_frame[upper_bed_border+10:lower_bed_border-10,0:500]
        depth_01 = ar_depth_frame[upper_bed_border+10:lower_bed_border-10,bed_width//2+50:bed_width]
        depth_bd = np.where((depth_01<1.2))
        up = False
        if(len(depth_bd[0])!=0):
            up = True
            row_min = np.min(depth_bd[0])+upper_bed_border+10
            row_max = np.max(depth_bd[0])+upper_bed_border+10
            col_min = np.min(depth_bd[1])+bed_width//2+50
            col_max = np.max(depth_bd[1])+bed_width//2+50
        return row_min,row_max,col_min,col_max,up



"""

def vertical_detected2(upper_bed_border, lower_bed_border, ar_depth_frame):
    row_min = row_max = col_min = col_max = np.zeros(4)
        
    border_row =  [upper_bed_border+10,upper_bed_border+10,(upper_bed_border+lower_bed_border)//2,(upper_bed_border+lower_bed_border)//2]
    border_col =  [0,0,bed_width//2,bed_width//2]
    depth  = depth1= [0,0,0,0]
    up = [False,False,False,False]
        
    depth01 = ar_depth_frame[upper_bed_border+10:(upper_bed_border+lower_bed_border)//2,0:bed_width//2]
    depth[1] = ar_depth_frame[upper_bed_border+10:(upper_bed_border+lower_bed_border)//2,bed_width//2:bed_width]
    depth[2] = ar_depth_frame[(upper_bed_border+lower_bed_border)//2:lower_bed_border-10,0:bed_width//2]    
    depth[3] = ar_depth_frame[(upper_bed_border+lower_bed_border)//2:lower_bed_border-10,bed_width//2:bed_width]
        
        
        
        
        
    for i in range(1):

            
        
        #depth1[i] = np.where((depth[i]<1.2))
        depth1 = np.where((depth01<1.2))
        
        if(len(depth1[0])!=0):
            up[i] = True
            row_min[i] = np.min(depth1[0])+border_row[i]
            row_max[i] = np.max(depth1[0])+border_row[i]
            col_min[i] = np.min(depth1[1])+border_col[i]
            col_max[i] = np.max(depth1[1])+border_col[i]
        
        
    return row_min,row_max,col_min,col_max,up

"""


try:
    start_time = time.time()
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, "meanwhile.bag")

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Start streaming from file
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
    # rs.align允许我们执行深度帧与其他帧的对齐
    # “align_to”是我们计划对齐深度帧的流类型。
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    while True:
    
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get depth frame
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_ndarray = np.asanyarray(depth_frame.get_data())*depth_scale
        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        start_time = time.time()
    
        #top_border1,top_border2,down_border1,down_border2 = get_bed_border(depth_frame)
        a, b, over_upper_border = upper_border_detected(131,depth_frame)
        c, d, over_lower_border = lower_border_detected(375,depth_frame)
        row_min_l,row_max_l,col_min_l,col_max_l,up_l = vertical_detected_left(131,375,depth_ndarray) 
        row_min_r,row_max_r,col_min_r,col_max_r,up_r = vertical_detected_right(131,375,depth_ndarray)

        color_image = np.asanyarray(color_frame.get_data())

        
        # Render image in opencv window
        green = (0, 255, 0)
        red = (255,0,0)
        # 画一条绿色 直线，从左上角到右下角
        cv2.line(color_image, (0, top_border1), (600, top_border2), green,3) 
        cv2.line(color_image, (0, down_border1), (600, down_border2), green,3) 
        #cv2.line(color_image, (0, a), (500, b), green,3) 
        #cv2.line(color_image, (0, c), (500, d), green,3) 
        if over_upper_border:
            cv2.line(color_image, (a,top_border1),(b,top_border2),red ,5)
        if over_lower_border:
            cv2.line(color_image, (c,down_border1),(d,down_border2),red ,5)
            
        point_color = (0, 0, 255) # BGR
        thickness = 4
        #if(up):
            #print(row_min,row_max,col_min,col_max)
        
        
        """
        for i in range(1):
            if(up[i]):

                print((col_min[i], row_min[i]),(col_max[i], row_max[i]))
                
                cv2.rectangle(color_image,
                            (col_min[i], row_min[i]),
                            (col_max[i], row_max[i]),
                            point_color, 2)
                
        """

        if(up_l):
            #cv2.rectangle(color_image, (col[0][0],col[1][0]), (col[0][len(col)-1],col[1][len(col)-1]), point_color, thickness, 4)
            
            cv2.rectangle(color_image,
                          (col_min_l, row_min_l),
                          (col_max_l, row_max_l),
                          point_color, 2)
               
        
        if(up_r):
            #cv2.rectangle(color_image, (col[0][0],col[1][0]), (col[0][len(col)-1],col[1][len(col)-1]), point_color, thickness, 4)
            
            cv2.rectangle(color_image,
                          (col_min_r, row_min_r),
                          (col_max_r, row_max_r),
                          point_color, 2)

         



        end_time = time.time()
        print((end_time - start_time)*1000)
         
        



        cv2.imshow("Depth Stream", depth_color_image)
        
        cv2.imshow("Color Stream", color_image) 
        #cv2.imshow('Color Stream',img2)


        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass

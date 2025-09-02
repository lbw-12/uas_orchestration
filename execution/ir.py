import os
import cv2
import rasterio
import numpy as np
import shutil
import argparse
import time   #PFR added time stuff

def bilateral_filter_cv(img, d=9, sigma_color=75, sigma_space=75):
    filtered_img = np.zeros_like(img)

    if img.dtype != np.uint8:
                img = img.astype(np.uint8)

    for i in range(3):
        filtered_img[:, :, i] = cv2.bilateralFilter(img[:, :, i], d, sigma_color, sigma_space)
    return filtered_img

def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def rgb_to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def convert_to_gis_coordinates(keypoints, transform, scale_percent):
    gis_coords = []
    for kp in keypoints:
        x, y = kp.pt
        x /= (scale_percent / 100.0)
        y /= (scale_percent / 100.0)
        gis_coord = transform * (x, y)
        gis_coords.append(gis_coord)
    return gis_coords

def process_image(target_image_path, reference_image_path, output_dir):
 
    print('MYINFO ---------------------------------------------------------')
    print('MYINFO, starting targpath:',target_image_path)
    start = time.time()

    with rasterio.open(target_image_path) as target_img:
        bounds = target_img.bounds   #PFR to target image is georefd,ref image is mosaic complete                                     
        original_transform = target_img.transform
        target_data  = target_img.read([1, 2, 3])

    with rasterio.open(reference_image_path) as reference_img:
        refbounds = reference_img.bounds   #PFR check these
        window = reference_img.window(*bounds)  #the orthomosaic and cut out stuff outside target bounds

        window_data = reference_img.read([1, 2, 3], window=window) #PFR try 1.5? )
        reference_transform = reference_img.window_transform(window)
        
        print(f"Bounds reading, windowing: {time.time()-start:.4f} seconds")
    
    #PFR added, check if windowdata shape is 0,0 for ht,width,which means raster-read 
    #fail to find pixels for reference imag in that window given by the target
    if window_data.shape[1]==0 or window_data.shape[2]==0:
      print("MYINFO, WARNING, in processing image, no pixels found in reference image within target's bounds")
      print('MYINFO, in proc image b4 stime,targbnds:',bounds)
      print('MYINFO, in proc image b4 stime,refbnds:',refbounds)
      print('MYINFO, in proc imag b4 stime hw of targ,ref data',target_data.shape,window_data.shape)
      print('MYINFO, WARNING, in processing image, skipping processing for this image: ')
      return
    else: #continue
      start = time.time()
      scale_percent = 50  # percentage of original size  
      target_data_resized = resize_image(target_data.transpose(1, 2, 0), scale_percent)
      window_data_resized = resize_image(window_data.transpose(1, 2, 0), scale_percent)

      #for debugging, original flight image (after geotagging)
      #if doflit:
      #  flight_data_resized = resize_image(flight_data.transpose(1, 2, 0), scale_percent)
    
      target_data_filtered = bilateral_filter_cv(target_data_resized)
      window_data_filtered = bilateral_filter_cv(window_data_resized)
    
      target_data_eq = equalize_histogram_color(target_data_filtered.astype('uint8'))
      window_data_eq = equalize_histogram_color(window_data_filtered.astype('uint8'))

      target_gray = rgb_to_gray(target_data_eq)
      window_gray = rgb_to_gray(window_data_eq)

      print(f"Preprocessing: {time.time()-start:.4f} seconds")
      start = time.time()
  
      # number of octaves = 4, number of scale levels = 5, initial 
      #sigma 1.6, k=sqrt(2)  as optimal values.

      sift = cv2.SIFT_create()
      kp1 = sift.detect(target_gray, None) 
      kp2 = sift.detect(window_gray, None)
    

      kp1 = sorted(kp1, key=lambda x: -x.response)[:10000]
      kp2 = sorted(kp2, key=lambda x: -x.response)[:10000]

      kp1, des1 = sift.compute(target_gray, kp1)
      kp2, des2 = sift.compute(window_gray, kp2)

      des1 = np.float32(des1)
      des2 = np.float32(des2)
    
      gis_coords_target = convert_to_gis_coordinates(kp1, original_transform, scale_percent)
      gis_coords_reference = convert_to_gis_coordinates(kp2, reference_transform, scale_percent)

      search_size = 2.0  # Boundary size in meter

      filtered_matches = []

      #PFR added for getting mean distances, 
      #    and filtering out sift points with feature distance > mean
      good_matches_distm=[]
      good_matches_distn=[]

      #PFR added track distance across all keypoints
      print(f"SIFT: {time.time()-start:.4f} seconds")

      #limit search to a box in geometric space
      start = time.time()
      for i, (t_gis, t_kp) in enumerate(zip(gis_coords_target, kp1)):
          min_x = t_gis[0] - search_size
          max_x = t_gis[0] + search_size
          min_y = t_gis[1] - search_size
          max_y = t_gis[1] + search_size

          #PFR, save list of k,j values in kp2 that are in this search area          
          region_kp      = [k for j, k in enumerate(kp2) if min_x <= gis_coords_reference[j][0] <= max_x and min_y <= gis_coords_reference[j][1] <= max_y]
          region_indices = [j for j, k in enumerate(kp2) if min_x <= gis_coords_reference[j][0] <= max_x and min_y <= gis_coords_reference[j][1] <= max_y]
                            
          if not region_kp or not region_indices:
              continue

          region_des = np.array([des2[j] for j in region_indices], dtype=np.float32)
        
          #now find matches in FEATURE space, for this ith-kp1 point,
          #      from the kp2 points in search region,
          if len(region_des) >= 2:
              FLANN_INDEX_KDTREE = 1
              index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
              search_params = dict(checks=50)

              flann = cv2.FlannBasedMatcher(index_params, search_params)
              matches = flann.knnMatch(des1[i].reshape(1, -1), region_des, k=2) #return k=2 matches

              #save matches in which the best match (m.distance) is 70% of 2nd best
              #   match (n.distance), where m,n are best,2nd best match
              good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
              filtered_matches.extend(good_matches)

              #PFR not save list of matches that are also not too far
              #so first look at same criteria, but save the distance values
              good_matches_distm.extend([m.distance for m, n in matches if m.distance < 0.7 * n.distance])
              good_matches_distn.extend([n.distance for m, n in matches if m.distance < 0.7 * n.distance])
              for match in good_matches:
                  match.trainIdx = region_indices[match.trainIdx]
                  match.queryIdx = i

      #PFR save this info
      num_filtmats=len(filtered_matches)

      #PFR extra filter to choose better half of feature matches
      if 1:
          filtered2=[]
          good_meandist=np.mean(np.asarray(good_matches_distm))
          for matches,distval in zip(filtered_matches,good_matches_distm):
              if distval<good_meandist:   #only use better set of matches
                  filtered2.append(matches)
          filtered_matches=filtered2
          num_filtmats=len(filtered_matches)
          print('MYINFO, num filtrd matches <mean:',len(filtered_matches))
      #end PFR extra filter -------------------

      print(f"Feature Matching: {time.time()-start:.4f} seconds")
      start = time.time()
      if len(filtered_matches) >= 20:  #20 is a heuristic from openCV examples

          src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2)
          dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2)

          src_pts *= (100.0 / scale_percent)  #above the image was resized down, now points are expanded 
          dst_pts *= (100.0 / scale_percent)

          matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.5)
        

          #=====================================================================
          #PFR added this sectin to score images according to how much the 
          # homography matrix has high values in the 3rd row (these correspond to
          # a skewed perspective when transforming the image)

          failth=0.00005  #a cutoff that indicates a failed homography (image is skipped)
          midth =0.00003; #a middle cutoff,image written to some extra folder

          z1,z2 = matrix[2,0:2] 
          if (abs(z1)>failth or abs(z2)>failth) :    #threshhold to say bad tranformatin perhaps
                    score='fail'
          elif (abs(z1) > midth or abs(z2)> midth) :    #threshhold to say bad tranformatin perhaps
                    score='mid'
          elif num_filtmats<50:
                    score='mid'
          else:     score='pass'
          scorevals="{:.7f}".format(z1)+'_'+"{:.7f}".format(z2)
          print('MYINFO, homography z elems:',target_image_path[-8:-1]+","+score+","+scorevals)

          #output matrix values also
          matvals=target_image_path+" , "+ reference_image_path+" , "+\
                       target_image_path[-8:-1]+" , " +str(num_filtmats)+", "+score+", "
          for mi in range(3):
            for mj in range(3):
                matvals=matvals+'{:3.5f} ,  '.format(matrix[mi,mj]) 
          print('MYINFO matvals:',matvals, ' 9999')
          #=====================================================================

          if matrix is not None:   
            aligned_image = cv2.warpPerspective(target_data.transpose(1, 2, 0), matrix,
                                                (window_data.shape[2], window_data.shape[1])).transpose(2, 0, 1)
            with rasterio.open(reference_image_path) as reference_img:
                new_meta = reference_img.meta.copy()
                new_meta.update({
                    'height': aligned_image.shape[1],
                    'width': aligned_image.shape[2],
                    'transform': reference_img.window_transform(window),
                    'count': 3
                })
            
            if score=='pass':
               output_path = os.path.join(output_dir, f'ir_{os.path.basename(target_image_path)}')
            elif score=='mid':
               #output_path = os.path.join(output_dir[0:-1]+"_MID", os.path.basename(target_image_path))
               output_path = os.path.join("/fs/ess/PAS2699/nitrogen/sandbox/IR_MID", os.path.basename(target_image_path))
            if score != 'fail' :
                with rasterio.open(output_path, 'w', **new_meta) as dest:
                      dest.write(aligned_image)
                print(f"Alignment completed for {os.path.basename(target_image_path)}.")
            else:
                print(f"Alignment failed for  {os.path.basename(target_image_path)}.")

          else:
            print(f"Homography matrix could not be computed for {os.path.basename(target_image_path)}.")
      else:
        print(f"Not enough good matches for {os.path.basename(target_image_path)}.")

      print(f"Homography Calculation and Registration: {time.time()-start:.4f} seconds")
      return True #PFR added this to just keep going

def process_images_for_date(input_dir, output_dir, reference_image_path):
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if not file.startswith('._'):
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
    else:
        os.makedirs(output_dir)
    
    processed_images = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".tif") and not filename.startswith("._"):
            target_image_path = os.path.join(input_dir, filename)
            success           = process_image(target_image_path, reference_image_path, output_dir)
            if success:
                processed_images.append(filename)
                print(f"Processed {filename}")
            else:
                print(f"Failed to process {target_image_path}. Trying the next image...")
    return processed_images


def main():
    parser = argparse.ArgumentParser(description="Process images for image registration")
    parser.add_argument("input_dir", type=str, help="Directory containing target images")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images")
    parser.add_argument("orthomosaic_image_path", type=str, help="Path to the orthomosaic image")

    args = parser.parse_args()
    print('MYINFO args',args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    processed_images = process_images_for_date(args.input_dir, args.output_dir, args.orthomosaic_image_path)
    print(f"Processed images: {len(processed_images)}")

if __name__ == "__main__":
    main()

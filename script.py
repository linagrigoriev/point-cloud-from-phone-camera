import cv2
import numpy as np
import glob
import os

def save_point_cloud_to_ply(points_3d, colors, filename):
    with open(filename, 'w') as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points_3d)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (x, y, z), color in zip(points_3d, colors):
            r, g, b = color
            f.write("{:.4f} {:.4f} {:.4f} {} {} {}\n".format(x, y, z, r, g, b))

def triangulate_points2(disparity_map, points, baseline, camera_matrix, image):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    u0 = camera_matrix[0, 2]
    v0 = camera_matrix[1, 2]

    points_3d = []
    colors = []

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, c = image.shape

    depth = 255 - disparity_map

    for v in range(h):
        for u in range(w):
            if disparity_map[v][u] > 0:
                # Z = depth[v][u]
                disparity = disparity_map[v][u]
                Z = fx * baseline / disparity_map[v][u]
                X = baseline * (u - u0) / disparity
                Y = baseline * (v - v0) * fx / (fy * disparity)
                # X = (u - u0) * Z / fx
                # Y = (v - v0) * Z / fy
                points_3d.append((X, Y, Z))
                color = image[v][u]
                colors.append(color)
    
    print(f"Total points: {len(points_3d)}")
    return points_3d, colors


# def triangulate_points(disparity_map, points, baseline, camera_matrix, image):

#     fx = camera_matrix[0, 0]
#     fy = camera_matrix[1, 1]

#     u0 = camera_matrix[0, 2]
#     v0 = camera_matrix[1, 2]

#     points_3d = []
#     colors = []
    

#     for i in range(len(points)):
#         pt1 = points[i]
        
#         disparity = disparity_map[int(pt1[1]), int(pt1[0])]

#         if disparity > 0:
#             Z = fx * baseline / disparity
#             X = (pt1[0] - u0) * Z / fx
#             Y = (pt1[1] - v0) * Z / fy

#             points_3d.append((X, Y, Z))
#             color = image[int(pt1[1]), int(pt1[0])]
#             colors.append(color)

#     print(f"Total points: {len(points_3d)}")
#     return points_3d, colors


def compute_disparity_map(img1_rectified, img2_rectified):

    img1_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                numDisparities=16, blockSize=15)

    disparity = stereo.compute(img1_gray, img2_gray)
    disparity = disparity

    disparity_normalized = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    print(f"Disparity range: min = {disparity_normalized.min()}, max = {disparity_normalized.max()}")


    # Display the disparity map
    # cv2.imshow("Disparity Map", disparity_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return disparity_normalized

def rectify_images(image1, image2, points1, points2, matches):

    fundamental_matrix, inliers = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    essential_matrix = cv2.findEssentialMat(points1, points2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]

    retval, R, t, mask = cv2.recoverPose(essential_matrix, points1, points2, camera_matrix)

    baseline = np.linalg.norm(t)

    inliers = inliers.ravel() # falatten the inliers

    # print(f"\nFundamental Matrix: {fundamental_matrix}")
    # print(f"\nInliers: {inliers}")

    retval, H1, H2 = cv2.stereoRectifyUncalibrated(points1[inliers == 1], points2[inliers == 1], fundamental_matrix, image1.shape[:2])


    if retval:
        img1_rectified = cv2.warpPerspective(image1, H1, (image1.shape[1], image1.shape[0]))
        img2_rectified = cv2.warpPerspective(image2, H2, (image2.shape[1], image2.shape[0]))

    return img1_rectified, img2_rectified, baseline


def feature_detection(image1, image2):

    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.equalizeHist(img1_gray)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.equalizeHist(img2_gray)

    image1_blur = cv2.GaussianBlur(img1_gray, (5, 5), 0)
    image2_blur = cv2.GaussianBlur(img2_gray, (5, 5), 0)

    # Automatic feature detection - ORB
    orb = cv2.ORB_create(nfeatures=100000)
    kp1, descriptors1 = orb.detectAndCompute(image1_blur, None)
    kp2, descriptors2 = orb.detectAndCompute(image2_blur, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Number of matches: {len(matches)}")

    points1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in matches])

    img_match = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Feature Matches", img_match)
    # cv2.waitKey(0)

    # points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    # points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    return img_match, points1, points2, matches


def camera_calibration(images_path, output_path, calibration_path, chessboard_size=(7,4), square_size=0.03):

    # Define World Coordinates (X) Corresponding to detected Corners (U)
    chessboard_coordinates = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    chessboard_coordinates[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    images = glob.glob(os.path.join(images_path, "*.jpg"))

    object_points = []
    image_points = []

    img_gray_shape = None

    for i in images:
        image = cv2.imread(i)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("img_gray", img_gray)
        # cv2.waitKey(0)

        retval, corners = cv2.findChessboardCorners(img_gray, chessboard_size) # retval = True
        print(retval)

        if retval:
            corners2 = cv2.cornerSubPix(
                img_gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            object_points.append(chessboard_coordinates)
            image_points.append(corners2)

            img_gray_shape = img_gray.shape[::-1]
        
        # else: 
        #     raise ValueError("No chessboard corners were detected in the provided images. Calibration cannot proceed.")

        cv2.drawChessboardCorners(image, chessboard_size, corners2, retval)
        cv2.imshow('Chessboard', image)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

    retval, camera_matrix, distorsion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        object_points, image_points, img_gray_shape, None, None
    )


    print(f"\nCamera Matrix: {camera_matrix}\n\nDistorsion Coefficients: {distorsion_coefficients}")

    for i in images:
        image = cv2.imread(i)
        img_undist = undistortion(camera_matrix, distorsion_coefficients, image)
        output_path_img = os.path.join(output_path, os.path.basename(i))
        cv2.imwrite(output_path_img, img_undist)

    return camera_matrix, distorsion_coefficients

def undistortion(camera_matrix, distorsion_coefficients, image):

    h, w = image.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distorsion_coefficients, (w, h), 1, (w, h)
    )

    img_undist = cv2.undistort(image, camera_matrix, distorsion_coefficients, None, new_camera_matrix)

    # Check if ROI is valid
    x, y, w, h = roi
    if w > 0 and h > 0:
        # Crop only if ROI is valid
        img_undist = img_undist[y:y+h, x:x+w]


    else:
        print("Invalid ROI, returning full undistorted image without cropping.")

    return img_undist

images_path="images2"
output_path="output"
calibration_path="calibration_data.npz"

camera_matrix, distorsion_coefficients = camera_calibration(images_path, output_path, calibration_path)

image1_path = "images3/scene5_0.jpg"
image2_path = "images3/scene5_1.jpg"

image1 = cv2.imread(image1_path)
# cv2.imshow('Undistorted Image', image1)
# cv2.waitKey(0)
image2 = cv2.imread(image2_path)

img1_undist = undistortion(camera_matrix, distorsion_coefficients, image1)
img2_undist = undistortion(camera_matrix, distorsion_coefficients, image2)

# cv2.imshow('Undistorted Image', img1_undist)
# cv2.waitKey(0)

img_match, points1, points2, matches = feature_detection(img1_undist, img2_undist)

img1_rectified, img2_rectified, baseline = rectify_images(img1_undist, img2_undist, points1, points2, matches)

output_path_img = "images3/image_match.jpg"
cv2.imwrite(output_path_img, img_match)

output_path_img = "images3/image1.jpg"
cv2.imwrite(output_path_img, img1_rectified)

output_path_img = "images3/image2.jpg"
cv2.imwrite(output_path_img, img2_rectified)

print(f"Baseline: {baseline}")
disparity_map = compute_disparity_map(img1_rectified, img2_rectified)

output_path_img = "images3/disparity_map.jpg"
cv2.imwrite(output_path_img, disparity_map)

points_3d, colors = triangulate_points2(disparity_map, points1, baseline, camera_matrix,img1_rectified)

ply_filename = "images3/point_cloud_5.ply"
save_point_cloud_to_ply(points_3d, colors, ply_filename)
print(f"Point cloud saved to {ply_filename}")

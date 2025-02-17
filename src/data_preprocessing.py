import glob
import os
import shutil

import cv2
import numpy as np

from utils.config import check_config_file, check_config_keys
from utils.preprocessing import preprocess


def aruco_extraction(img):
    """
    Extracts the aruco signs from the given image, and selects the boundaries points of the form

    Args:
        img (numpy.ndarray): an image from the dataset

    Returns:
        numpy.ndarray or None: boundaries of the form
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Detect the markers in the image
    

    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(
        img
    )

    # Checks how many markers are detected
    if len(marker_corners) != 4:
        print("{} arucos detected instead of 4!".format(len(marker_corners)))
        return None

    # flatten the marker_corners array
    marker_corners = [mc[0] for mc in marker_corners]

    # corners based on the ids [30: top left, 31:top right, 33:bottom right, 32:bottom left]

    # selects the boundaries clock wise(top left point of the top left marker,
    #                                   top right point of the top right marker,
    #                                   bottom right point of the bottom right marker,
    #                                   bottom left point of the bottom left marker)
    boundaries = np.array(
        [
            marker_corners[int(np.where(marker_ids == 30)[0])][0],
            marker_corners[int(np.where(marker_ids == 31)[0])][1],
            marker_corners[int(np.where(marker_ids == 33)[0])][2],
            marker_corners[int(np.where(marker_ids == 32)[0])][3],
        ],
        dtype=np.float32,
    )
    return boundaries


def form_extraction(img, corners, form_width, form_height):
    """
    Applies perspective to the image and extracts the form

    Args:
        img (numpy.ndarray): an image from the dataset
        corners (numpy.ndarray): position of the corners of the form
        form_width (int): width of the form
        form_height (int): height of the form

    Returns:
        numpy.ndarray: image of the extracted form
    """
    form_points = np.array(
        [(0, 0), (form_width, 0), (form_width, form_height), (0, form_height)]
    ).astype(np.float32)

    # applies perspective tranformation
    perspective_transformation = cv2.getPerspectiveTransform(corners, form_points)
    form = cv2.warpPerspective(
        img, perspective_transformation, (form_width, form_height)
    )
    return form


def cell_extraction(
    img,
    img_path,
    extracted_dataset_path,
    type,
    form_width,
    form_height,
    cell_width,
    cell_height,
    gaussian_kernel
):
    """
    Extracts cells and the saves them based on the type of the given form

    Args:
        img (numpy.ndarray): an image from the dataset
        img_path (str): path of the image
        extracted_dataset_path (str): path of the directory for the final data
        type (str): type of the form, either 'a' or 'b'
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
        gaussian_kernel (int): the gaussian kernel
    """
    # Calculate cell dimensions based on the number of lines
    num_horizontal_lines = 21
    num_vertical_lines = 14
    cell_width = form_width // num_vertical_lines
    cell_height = form_height // num_horizontal_lines
    for row in range(num_horizontal_lines):
        if type == "a":  # the directory is named for the form 'a'
            if row < 2:  # cells for number '0' and '1'
                directory = str(row)
            elif row > 18:  # cells for number '2' and '3'
                directory = str(row - 17)
            else:
                directory = str(row + 8)  # cells for the first part of the letters

        elif type == "b":  # the directory is named for the form 'b'
            if row < 2:  # cells for number '3' and '4'
                directory = str(row + 4)
            elif row > 16:  # cells for number '6', '7', '8', and '9'
                directory = str(row - 11)
            else:  # cells for the second part of the letters
                directory = str(row + 25)
        for col in range(num_vertical_lines):
            # calculates the position of the cells
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            cell = img[y1+7:y2-7, x1+7:x2-7]
            
            # drops the cells that contain markers
            if row < 2 and col < 2:
                continue
            if row < 2 and col > 11:
                continue
            if row > 18 and col < 2:
                continue
            if row > 18 and col > 11:
                continue

            cell = preprocess(cell, gaussian_kernel)
            if cell is not None:
                cv2.imwrite(
                    extracted_dataset_path
                    + "/"
                    + directory
                    + "/"
                    + img_path[img_path.find("/" + type + "\\") + 3 : -4]
                    + "_"
                    + str(col)
                    + ".jpg",
                    cell,
                )


def make_directories(path, num_classes):
    """
    Creates directory  and the needed subdirectories. 0-9 are correspondent to numbers and 10-41 are correspondent to the persian letters.

    Args:
        path (str): The name of the main directory for creating the directories
        num_classes (int): indicates the number of claases
    """
    # Creating the main directory
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")

    # Creating the subdirectories within the main directory
    for i in range(num_classes):
        if not os.path.exists(path + "/" + str(i)):
            os.makedirs(path + "/" + str(i))

def remove_grid_from_image(image, form_width, form_height):
    """
    removes the grid as a preprocessing from the forms and makes the data more clean 
    despite the error in detecting lines by Hough Transform Line Detection Method

    Args:
        image: the image that is read using opencv(cv2)
        form_width: the width of a typical form
        form_height: the hight of a typical form
    """
    filter = True

    img = image

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,90,150,apertureSize = 3)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.erode(edges,kernel,iterations = 1)
    
    cv2.imwrite("canny.jpg", edges)
    lines2 = cv2.HoughLines(edges,1,np.pi/180,150)
    lines3 = []
    epsilon = np.pi/360.0
    
    if not np.array(lines2).any():
        print('No lines were found')
        return image
    
    num_horizontal_lines = 21
    num_vertical_lines = 14
    cell_width = form_width // num_vertical_lines
    cell_height = form_height // num_horizontal_lines
    lineapproxrhoh = np.array([cell_width*i for i in range(num_vertical_lines)])
    lineapproxrhov = np.array([cell_height*j for j in range(num_horizontal_lines)])
    for line in lines2:
        rho, theta = line[0]
        
        if abs(theta - np.pi/2) < epsilon and np.any(abs(rho-lineapproxrhov) < 5):
            lines3.append(line)
        if abs(theta - 0) < epsilon and np.any(abs(rho-lineapproxrhoh) < 5):
            lines3.append(line)
    lines = np.array(lines3)


    if filter:
        rho_threshold = 32
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]: # and only if we have not disregarded them already
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)): # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)

    return img

def label_dataset(
    dataset_path,
    labeled_dataset_path,
    type,
    form_width,
    form_height,
    cell_width,
    cell_height,
    gaussian_kernel
):
    """labels dataset by extracting form and each cell and saving them to a folder that represents the class of each cell

    Args:
        dataset_path (str): path of the dataset
        labeled_dataset_path (str): path for storing the labeled dataset
        type (str): type of the form, either 'a' or 'b'
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
        gaussian_kernel (int): gaussian kernel
    """

    for image_path in glob.glob(dataset_path + "/" + type + "/*.*"):
        image = cv2.imread(image_path)
        
        corners = aruco_extraction(image)
        if corners is None:
            print(
                f"The image {image_path[image_path.find('/' + type + '/')+3:]} is dropped."
            )
            continue
        form = form_extraction(image, corners, form_width, form_height)
        form2 = remove_grid_from_image(form, form_width, form_height)
        
        cell_extraction(
            form2,
            image_path,
            labeled_dataset_path,
            type,
            form_width,
            form_height,
            cell_width,
            cell_height,
            gaussian_kernel
        )


def finalise_dataset(
    labeled_dataset_path, final_dataset_path, num_classes, val_ratio, test_ratio
):
    """
    Splits dataset into train, val, and test set based on the given ratio
    Args:
        labeled_dataset_path (str): path of the labeled dataset
        final_dataset_path (str): path for finalised dataset
        num_classes (int): number of classes
        val_ratio (float): ratio for validation set
        test_ratio (float): ratio for test set
    """
    make_directories(final_dataset_path + "/train", num_classes)
    make_directories(final_dataset_path + "/val", num_classes)
    make_directories(final_dataset_path + "/test", num_classes)

    for cls in range(num_classes):
        all_file_names = os.listdir(labeled_dataset_path + "/" + str(cls))

        np.random.shuffle(all_file_names)
        train_file_names, val_file_names, test_file_names = np.split(
            np.array(all_file_names),
            [
                int(len(all_file_names) * (1 - val_ratio + test_ratio)),
                int(len(all_file_names) * (1 - test_ratio)),
            ],
        )

        train_file_names = [
            labeled_dataset_path + "/" + str(cls) + "/" + name
            for name in train_file_names.tolist()
        ]
        val_file_names = [
            labeled_dataset_path + "/" + str(cls) + "/" + name
            for name in val_file_names.tolist()
        ]
        test_file_names = [
            labeled_dataset_path + "/" + str(cls) + "/" + name
            for name in test_file_names.tolist()
        ]

        print("________________________")
        print("Class : ", cls)
        print("Total images: ", len(all_file_names))
        print("Training: ", len(train_file_names))
        print("Validation: ", len(val_file_names))
        print("Testing: ", len(test_file_names))

        # Copy-pasting images
        for name in train_file_names:
            shutil.copy(name, final_dataset_path + "/train/" + str(cls))

        for name in val_file_names:
            shutil.copy(name, final_dataset_path + "/val/" + str(cls))

        for name in test_file_names:
            shutil.copy(name, final_dataset_path + "/test/" + str(cls))


def preproessing(config):
    """preprocess data

    Args:
        config (dict): the config file
    """
    required_keys = [
        "dataset.splitted",
        "dataset.labeled",
        "dataset.final",
        "pre_processing.form_width",
        "pre_processing.form_height",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
        "pre_processing.num_classes",
        "pre_processing.val_ratio",
        "pre_processing.test_ratio",
    ]

    check_config_keys(config, required_keys)

    form_width = config["pre_processing"].get("form_width")
    form_height = config["pre_processing"].get("form_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    num_classes = config["pre_processing"].get("num_classes")

    gaussian_kernel = config["pre_processing"].get("gaussian_kernel")

    val_ratio = config["pre_processing"].get("val_ratio")
    test_ratio = config["pre_processing"].get("test_ratio")

    dataset_path = config["dataset"].get("splitted")

    labeled_dataset_path = config["dataset"].get("labeled")
    final_dataset_path = config["dataset"].get("final")

    make_directories(labeled_dataset_path, num_classes)

    # Extracting the forms of the type 'a'
    label_dataset(
        dataset_path,
        labeled_dataset_path,
        "a",
        form_width,
        form_height,
        cell_width,
        cell_height,
        gaussian_kernel
    )

    # Extracting the forms of the type 'b'
    label_dataset(
        dataset_path,
        labeled_dataset_path,
        "b",
        form_width,
        form_height,
        cell_width,
        cell_height,
        gaussian_kernel
    )

    print("dataset is extracted and labeled successfully.")

    finalise_dataset(
        labeled_dataset_path, final_dataset_path, num_classes, val_ratio, test_ratio
    )

    print("dataset is split to train, val, and test successfully.")


if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    preproessing(config)

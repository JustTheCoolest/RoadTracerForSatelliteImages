import cv2
import numpy as np

# Calibration constants
DIRECTION_SCALE_FACTOR = 0.01
LINE_THICKNESS = 1
LINSPACE_PRECISION = 100

# Load the image
img = cv2.imread('/home/goto/Dumpyard/vit1.jpg')

# Create a copy of the original image
img_copy = img.copy()

# Initialize a counter and variables to store the previous coordinates
counter = 0
prev_x, prev_y = -1, -1

# Calculate the diagonal length of the image
diagonal_length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

def calculate_direction_vector(x, y, prev_x, prev_y, diagonal_length):
    direction_vector = np.array([x - prev_x, y - prev_y])
    direction_length = np.linalg.norm(direction_vector)
    if direction_length == 0:
        return np.array([0, 0])
    scale_factor = DIRECTION_SCALE_FACTOR * diagonal_length / direction_length
    return (direction_vector * scale_factor).astype(int)

def calculate_perpendicular_vectors(direction_vector):
    return np.array([direction_vector[1], -direction_vector[0]]), np.array([-direction_vector[1], direction_vector[0]])

def calculate_division_points(prev_x, prev_y, perp_vector):
    return [np.array([prev_x, prev_y]) + i/4 * perp_vector for i in range(5)]

def calculate_corresponding_points(division_points, direction_vector):
    return [point + direction_vector for point in division_points]

def calculate_lines(division_points1, corresponding_points1, division_points2, corresponding_points2):
    lines = []
    for i in range(5):
        lines.append([division_points1[i], corresponding_points1[i]])
        lines.append([division_points2[i], corresponding_points2[i]])
    return lines

def create_rectangles(division_points, corresponding_points):
    n = min(len(division_points), len(corresponding_points))
    rectangles = []
    for i in range(n-1):
        rectangles.append([division_points[i], corresponding_points[i+1]])
    return rectangles

def create_rectangles_from_lines(lines):
    rectangles = []
    for i in range(0, len(lines), 2):
        line1 = lines[i]
        line2 = lines[i+1]
        rectangle = [line1[0], line1[1], line2[1], line2[0]]
        rectangles.append(rectangle)
    return rectangles

def is_point_in_rectangle(point, rectangle):
    pass

def iterate_rectangle_fill(rectangle, function):

    # Create a linspace from rectangle[0] to rectangle[1]
    linspace = np.linspace(rectangle[0], rectangle[1], num=LINSPACE_PRECISION)

    perpendicular_vector = rectangle[3] - rectangle[0]
    perpendicular_linspace = np.linspace([0, 0], perpendicular_vector, num=LINSPACE_PRECISION)

    for i in linspace:
        for j in perpendicular_linspace:
            function(i + j)

def test_function(point):
    # Change the color of the point to red in the image
    cv2.circle(img, point, radius=0, color=(0, 0, 255), thickness=-1)

def calulate_average_rgb(rectangle, function):
    pass

def click_event(event, x, y, flags, param):
    global counter, prev_x, prev_y
    if event == cv2.EVENT_LBUTTONDOWN:

        if counter % 2 == 0:
            # Restore the image to its original state
            img[:] = img_copy[:]
            # Draw a circle at the clicked location
            cv2.circle(img, (x, y), radius=LINE_THICKNESS*5//2, color=(0, 255, 0), thickness=-1)
            prev_x, prev_y = x, y
        else:
            # Create a point from prev_x and prev_y
            point = np.array([prev_x, prev_y])

            # Calculate the direction vector
            direction_vector = calculate_direction_vector(x, y, prev_x, prev_y, diagonal_length)

            # Draw a line from the previous coordinates to the current ones
            cv2.line(img, (prev_x, prev_y), np.array([prev_x, prev_y]) + direction_vector, color=(0, 255, 0), thickness=LINE_THICKNESS)

            # Calculate the perpendicular vectors
            perp_vector1, perp_vector2 = calculate_perpendicular_vectors(direction_vector)

            # Draw the perpendicular lines
            cv2.line(img, (prev_x, prev_y), tuple(np.array([prev_x, prev_y]) + perp_vector1), color=(0, 255, 0), thickness=LINE_THICKNESS)
            cv2.line(img, (prev_x, prev_y), tuple(np.array([prev_x, prev_y]) + perp_vector2), color=(0, 255, 0), thickness=LINE_THICKNESS)
            
            # Calculate the points that divide the perpendiculars into 4 parts
            division_points1 = calculate_division_points(prev_x, prev_y, perp_vector1)
            division_points2 = calculate_division_points(prev_x, prev_y, perp_vector2)

            # Calculate the corresponding points on the direction line
            corresponding_points1 = calculate_corresponding_points(division_points1, direction_vector)
            corresponding_points2 = calculate_corresponding_points(division_points2, direction_vector)

            # Extend the division_points1 array with the inverted division_points2 array
            unified_division_points = np.concatenate((division_points1[::-1][:-1], division_points2))

            # Extend the corresponding_points1 array with the inverted corresponding_points2 array
            unified_corresponding_points = np.concatenate((corresponding_points1[::-1][:-1], corresponding_points2[::-1]))

            # # Print the division points arrays
            # print("Division points 1:", division_points1)
            # print("Division points 2:", division_points2)

            # # Print the corresponding points arrays
            # print("Corresponding points 1:", corresponding_points1)
            # print("Corresponding points 2:", corresponding_points2)

            # # Print the unified division points array
            # print("Unified division points:", unified_division_points)

            # # Print the unified corresponding points array
            # print("Unified corresponding points:", unified_corresponding_points)

            # Store the points as an array of 8 lines
            lines = calculate_lines(division_points1, corresponding_points1, division_points2, corresponding_points2)

            # Draw the parallel lines
            for line in lines:
                cv2.line(img, tuple(line[0].astype(int)), tuple(line[1].astype(int)), color=(0, 255, 0), thickness=LINE_THICKNESS)
            
            # Create rectangles from lines
            rectangles = create_rectangles_from_lines(lines)

            # Print the point
            print("Point:", point)

            # Print the direction vector
            print("Direction vector:", direction_vector)

            # Print the rectangles
            for i, rectangle in enumerate(rectangles):
                print(f"Rectangle {i+1}:", rectangle)

            # In the click_event function, calculate the average RGB values for each rectangle
            # for rectangle in rectangles:
            #     iterate_rectangle_fill(rectangle, test_function)

        # Update the image display
        cv2.imshow('image', img)

        # Increment the counter
        counter += 1
             
# Create a window and set the callback
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_event)

# Display the image (this will block until the window is closed)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

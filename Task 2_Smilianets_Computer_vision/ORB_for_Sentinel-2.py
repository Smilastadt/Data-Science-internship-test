import cv2
import matplotlib.pyplot as plt


def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_image(image, title="Image"):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Function to detect and match keypoints using ORB
def detect_and_match_keypoints(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Initialize the ORB detector
    orb = cv2.ORB_create(nfeatures=500, edgeThreshold=15, patchSize=31, WTA_K=2)

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Apply ratio test
    good_matches = []
    for match in matches:
        if match.distance < 0.75 * match.trainIdx:
            good_matches.append(match)

    return kp1, kp2, good_matches


def visualize_keypoints_matches(image1, image2, kp1, kp2, matches):
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title("Keypoint Matches")
    plt.axis('off')
    plt.show()


def main():
    print("Welcome to the Image Matching Console!")
    print("1. Open Image 1")
    print("2. Open Image 2")
    print("3. Match KeyPoints")
    print("4. Exit")

    image1 = None
    image2 = None

    while True:
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            image_path = input("Enter the path of Image 1: ")
            image1 = read_image(image_path)
            visualize_image(image1, title="Image 1")

        elif choice == '2':
            image_path = input("Enter the path of Image 2: ")
            image2 = read_image(image_path)
            visualize_image(image2, title="Image 2")

        elif choice == '3':
            if image1 is not None and image2 is not None:
                kp1, kp2, matches = detect_and_match_keypoints(image1, image2)
                visualize_keypoints_matches(image1, image2, kp1, kp2, matches)
            else:
                print("Please open both images before matching keypoints.")

        elif choice == '4':
            print("Exiting the Image Matching Console.")
            break

        else:
            print("Invalid choice. Please enter a valid option (1/2/3/4).")


if __name__ == "__main__":
    main()

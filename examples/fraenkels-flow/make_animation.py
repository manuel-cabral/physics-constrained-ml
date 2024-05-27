import cv2
import os
import imageio.v2 as imageio

kind = 'gif'

#! GIF
if kind == 'gif':
    filenames = [f'src/frames/frame_{idx:03d}.png' for idx in range(0, 160)]
    # print(filenames)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('src/sampling_transformation.gif', images)

#! VIDEO
else:
    image_folder = 'src/frames/'
    video_name = 'src/sampling_transformation_input.avi'

    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = [img for img in sorted(os.listdir(image_folder))]

    # images = [f'{image_folder}/frame_{idx:03d}.png' for idx in range(0, 160)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps=10, frameSize=(width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
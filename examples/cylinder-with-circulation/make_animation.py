import imageio.v2 as imageio

filenames = [f'src/frames/frame_{i}.png' for i in range(0, 150)]
# print(filenames)

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('src/movie_base_16_2_new.gif', images)
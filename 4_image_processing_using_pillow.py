from PIL import Image

img=Image.open('images/nature.jpg')
print(img.format)
print(img.size) #(width,height)
print(img.mode)
small_img=img.resize((200,300)) #it didn't keep the aspect ratio, it just cropped it
small_img.save('images/nature_small.jpg')
img.thumbnail((200,300)) # it inplace resized to 200,300 but keep aspect ratio,
# so final size=200,133 and looks the original image. 300 does not matter here
img.save('images/nature_tmbnail.jpg')
print(img.size)

# crop image

cropped_img= img.crop((0,0,300,250)) #(x1,y1,x2,y2)
cropped_img.save('images/nature_cropped.jpg')

img2=Image.open('images/monkey.jpg')
img1=Image.open('images/nature.jpg')
img1.paste(img2,(20,30))
img1.save('images/pasted_image.jpg')

# rotation

img90=img2.rotate(90,expand=True) #expand=true means keepening the full image so that
# after rotation, image will not be cropped
img90.save('images/monkey_rotated.jpg')

# flip

# left right flip

img_flipLR=img2.transpose(Image.FLIP_LEFT_RIGHT)
img_flipLR.save('images/monkey_flipLR.jpg')


img_flipTB=img2.transpose(Image.FLIP_TOP_BOTTOM)
img_flipTB.save('images/monkey_flipTB.jpg')

# convert to gray image

gray_img=img2.convert('L') # replace L with RGB to convert to RGB
gray_img.save('images/monkey_grayscale.png',"png")


from PIL import Image
import os

# def resize_image_proportionally(input_image_path, output_image_path, max_size=672):
#     with Image.open(input_image_path) as img:
#         # 获取图片的原始宽和高
#         width, height = img.size

#         # 判断是否需要缩放（只有当图片的最小边长大于672时才缩放）
#         if max(width, height) > max_size:
#             # 保持宽高比计算新的宽度和高度
#             if width > height:
#                 new_height = max_size
#                 new_width = int(max_size * (width / height))
#             else:
#                 new_width = max_size
#                 new_height = int(max_size * (height / width))

#             # 缩放图片
#             img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
#             # 保存缩放后的图片
#             img_resized.save(output_image_path)
#         else:
#             # 如果不需要缩放，则直接保存原图
#             img.save(output_image_path)
def resize_image_proportionally(input_image_path, output_image_path, max_size=672):
    with Image.open(input_image_path) as img:
        width, height = img.size
        # 计算宽高比
        ratio = min(max_size/width, max_size/height)
        
        # 保持宽高比调整图片大小
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 缩放图片
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_image_path)



input_folder = "/mnt/d/Mocheg-main/Mocheg-main/data/test/images"
output_folder = "/mnt/d/Mocheg-main/Mocheg-main/data/test/resized_images"
for filename in os.listdir(input_folder):
        # 构建完整的文件路径
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    resize_image_proportionally(input_path, output_path)
import json

path = "./documents/train_dataset/train.json"
dir = "/home/ie/桌面/ZDW/projects/traffic_light/documents/train_dataset/train_images/"
file_write_obj = open("./dataset/train_dataset.txt", 'w')  # 新文件
last_name = 'oriname'
num = 0
with open(path,"r") as f:
    files = json.load(f)
    annotations = files["annotations"]
    for each_image in annotations:
        if each_image["inbox"] != []:
            label = -1
            label_name = dir + each_image["filename"].split("\\")[-1]
            obj = each_image["inbox"][0]
            label_class = obj["color"]
            print(label_name,label_class)
            if label_class == "red":
                label = 0
            elif label_class == "yellow":
                label = 1
            elif label_class == "green":
                label = 2
            label_box = ','.join(str(int(obj["bndbox"][i])) for i in obj["bndbox"])
            each_object = " "+str(label_box)+","+str(label)
            if label_name != last_name:
                file_write_obj.write('\n')
                file_write_obj.write(str(label_name)+each_object)
            else:
                file_write_obj.write(each_object)
            last_name = label_name
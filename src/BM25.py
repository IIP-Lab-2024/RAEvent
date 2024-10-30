import json

# 假设你的JSON文件的路径是 'input.json'
input_file_path = 'src/result/PromptCase/2stage_lecard_0411-072628.json'
output_file_path = 'src/result/event/PromptCase2.json'

# 读取原始JSON文件
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 转换数据格式
converted_data = {int(key): [int(item) for item in value] for key, value in data.items()}

# 将转换后的数据写入新的JSON文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(converted_data, file, ensure_ascii=False, indent=4)

print(f'Data has been converted and saved to {output_file_path}')

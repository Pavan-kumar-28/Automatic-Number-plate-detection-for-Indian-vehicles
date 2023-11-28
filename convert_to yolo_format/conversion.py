import os

# Replace these paths with your source folder
folder_path = 'C:\\Users\\NableIT02\\Desktop\\VDS\\Dataset\\test\\labels'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through all text files in the folder
for file_name in files:
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)
        
        # Read the contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Replace class index '0' with '1' in each line
        updated_lines = [line.replace('1 ', '0 ') for line in lines]
        
        # Write the updated lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)
        
        print(f"Updated {file_name}")

print("Replacement completed.")

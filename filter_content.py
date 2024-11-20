# 生成content/.gitignore文件，只包含draft: false的.md文件

import os  
import re  
  
def find_draft_false_files(directory):  
    draft_false_files = []  
    yaml_pattern = re.compile(r'^---\s*\n(.*?)^---\s*\n', re.DOTALL | re.MULTILINE)  # https://stackoverflow.com/questions/41620093/whats-the-difference-between-re-dotall-and-re-multiline

    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.md'):  
                file_path = os.path.join(root, file)  
                relative_path = os.path.relpath(file_path, directory)  
                  
                try:  
                    with open(file_path, 'r', encoding='utf-8') as f:  
                        content = f.read()  
                        match = yaml_pattern.search(content) 
                        if match:  
                            yaml_content = match.group(1)  
                            lines = yaml_content.strip().split('\n')  
                            draft_line = next((line for line in lines if 'draft:' in line), None)  
                            if draft_line and draft_line.strip().endswith('false'):  
                                draft_false_files.append(relative_path)  
                except Exception as e:  
                    print(f"Error processing file {file_path}: {e}")  
      
    return draft_false_files  
  
def write_gitignore(files):  
    gitignore_path = os.path.join(content_dir, '.gitignore')
    with open( gitignore_path , 'w', encoding='utf-8') as f:  
        f.write('/**/*.md\n')  
        f.write('\n')  
        for file in sorted(files):  
            s = file.replace('\\','/')
            f.write(f'!{s}\n')  
    print(f"Updated {gitignore_path}")
  
# 使用示例  
content_dir = 'content'  
draft_false_files = find_draft_false_files(content_dir)  

write_gitignore(draft_false_files)  


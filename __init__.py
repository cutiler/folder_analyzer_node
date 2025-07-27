import os
import json
from PIL import Image
import numpy as np
import torch

# --- FolderAnalyzerNode 클래스 ---
class FolderAnalyzerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("subfolder_list", "image_counts",)
    FUNCTION = "analyze_folder"
    CATEGORY = "Folder Tools"

    def analyze_folder(self, input_folder):
        subfolders = []
        image_counts = {}

        if not os.path.isdir(input_folder):
            print(f"Error: 입력 폴더가 존재하지 않거나 디렉토리가 아닙니다: {input_folder}")
            return ("", "")

        for root, dirs, files in os.walk(input_folder):
            for dir_name in dirs:
                subfolder_path = os.path.join(root, dir_name)
                subfolders.append(os.path.abspath(subfolder_path))

                count = 0
                try:
                    for entry in os.scandir(subfolder_path):
                        if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            count += 1
                except PermissionError:
                    print(f"권한 오류: {subfolder_path} 폴더에 접근할 수 없습니다.")
                    count = -1
                image_counts[os.path.abspath(subfolder_path)] = count

        subfolder_list_str = ",".join(subfolders)
        image_counts_str = ",".join([f"{k}:{v}" for k, v in image_counts.items()])

        # 파일 저장
        node_dir = os.path.dirname(os.path.abspath(__file__))
        output_filename = "folder_analysis_results.json"
        output_filepath = os.path.join(node_dir, output_filename)

        results = {
            "input_folder": os.path.abspath(input_folder),
            "subfolders": sorted(subfolders),
            "image_counts": image_counts
        }

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"분석 결과 저장됨: {output_filepath}")
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")

        return (subfolder_list_str, image_counts_str)

# --- ImageLoaderFromAnalyzedFolder 클래스 ---
class ImageLoaderFromAnalyzedFolder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        def get_subfolder_options():
            try:
                node_dir = os.path.dirname(os.path.abspath(__file__))
                analysis_filepath = os.path.join(node_dir, "folder_analysis_results.json")

                if os.path.exists(analysis_filepath):
                    with open(analysis_filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        subfolders = data.get("subfolders", [])
                        image_counts = data.get("image_counts", {})
                        
                        if subfolders:
                            options = []
                            for folder_path in subfolders:
                                count = image_counts.get(folder_path, 0)
                                display_name = f"{os.path.basename(folder_path)} (이미지: {count}개)"
                                options.append(display_name)
                            return options
            except Exception as e:
                print(f"분석 파일 로드 중 오류: {e}")
            
            return ["분석된 폴더 없음 - Folder Analyzer를 먼저 실행하세요"]

        return {
            "required": {
                "selected_folder": (get_subfolder_options(),),
                "current_index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "auto_increment": ("BOOLEAN", {"default": True})
            },
        }

    @classmethod 
    def IS_CHANGED(cls, **kwargs):
        node_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_filepath = os.path.join(node_dir, "folder_analysis_results.json")
        
        if os.path.exists(analysis_filepath):
            return str(os.path.getmtime(analysis_filepath))
        return ""

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("image", "image_filename", "image_filepath", "next_index",)
    FUNCTION = "load_image"
    CATEGORY = "Folder Tools"

    def load_image(self, selected_folder, current_index, auto_increment):
        # JSON 파일에서 실제 폴더 경로 찾기
        node_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_filepath = os.path.join(node_dir, "folder_analysis_results.json")
        
        selected_subfolder = None
        
        if os.path.exists(analysis_filepath):
            try:
                with open(analysis_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    subfolders = data.get("subfolders", [])
                    
                    # 선택된 폴더명으로 실제 경로 찾기
                    folder_basename = selected_folder.split(" (이미지:")[0].strip()
                    for folder_path in subfolders:
                        if os.path.basename(folder_path) == folder_basename:
                            selected_subfolder = folder_path
                            break
            except Exception as e:
                print(f"분석 파일 로드 중 오류: {e}")

        if not selected_subfolder or not os.path.isdir(selected_subfolder):
            print(f"Error: 선택된 서브폴더가 존재하지 않습니다: {selected_subfolder}")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", "", current_index)

        # 이미지 파일 목록 가져오기
        image_files = sorted([
            f for f in os.listdir(selected_subfolder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')) 
            and os.path.isfile(os.path.join(selected_subfolder, f))
        ])

        if not image_files:
            print(f"선택된 서브폴더에 이미지가 없습니다: {selected_subfolder}")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", "", current_index)

        # 현재 인덱스로 이미지 로드
        idx_to_load = current_index % len(image_files)
        image_filename = image_files[idx_to_load]
        image_filepath = os.path.join(selected_subfolder, image_filename)

        try:
            img = Image.open(image_filepath)
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]

            print(f"이미지 로드됨: {image_filepath} (인덱스: {idx_to_load}/{len(image_files)-1})")
            
            # 다음 인덱스 계산
            if auto_increment:
                next_index = (current_index + 1) % len(image_files)
            else:
                next_index = current_index
            
            return (img_tensor, image_filename, image_filepath, next_index)

        except Exception as e:
            print(f"이미지 로드 중 오류: {image_filepath}: {e}")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", "", current_index)

# ComfyUI 노드 매핑
NODE_CLASS_MAPPINGS = {
    "FolderAnalyzerNode": FolderAnalyzerNode,
    "ImageLoaderFromAnalyzedFolder": ImageLoaderFromAnalyzedFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderAnalyzerNode": "Folder Analyzer",
    "ImageLoaderFromAnalyzedFolder": "Image Loader (Folder Analyzer)"
}
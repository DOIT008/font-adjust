import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
import argparse
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Scale, Offset as offset
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
font_tools_available = True


class FontRendererAdjuster:
    """
    将SF Mono字体的渲染效果调整为与JetBrains Mono字体相似的风格，
    保持字符形状不变，但调整粗细、清晰度和整齐度
    """
    
    def __init__(self, sf_mono_path, jetbrains_mono_path=None, output_font_name="adjusted_sf_mono"):
        """
        初始化字体渲染调整器
        
        参数:
            sf_mono_path: SF Mono字体文件路径
            jetbrains_mono_path: JetBrains Mono字体文件路径（可选）
            output_font_name: 输出字体名称
        """
        self.sf_mono_path = sf_mono_path
        self.jetbrains_mono_path = jetbrains_mono_path
        self.output_font_name = output_font_name
        
        # JetBrains Mono风格特征参数
        # 字体渲染优化参数 - 经过调整以获得更清晰、整齐和美观的效果
        # 参数最佳实践说明：
        # - weight_adjustment: 1.2 提供更强的视觉存在感，同时保持细节清晰
        # - sharpness_factor: 1.3 增强边缘清晰度，使字体在各种分辨率下更易读
        # - spacing_adjustment: 1 适度收紧字符间距，提高代码可读性
        # - contrast_boost: 1.25 提高对比度，使字符在各种背景下更突出
        # - anti_aliasing_level: 3 提供更好的平滑效果，减少锯齿感
        self.jetbrains_mono_style = {
            "weight_adjustment":     1.2,  # 字重调整：增强线条强度
            "sharpness_factor":      1.3,  # 清晰度：提高边缘锐利度
            "spacing_adjustment":    1.05, # 字间距：优化字符间距离
            "contrast_boost":        1.25, # 对比度：增强黑白色调差异
            "anti_aliasing_level":   3     # 抗锯齿：平滑字体边缘
        }
        
        # 额外的字体美学优化参数
        self.font_aesthetics = {
            "crisp_rendering": True,       # 启用清晰渲染模式
            "balanced_proportions": True,  # 保持平衡的字符比例
            "enhanced_legibility": True    # 优化易读性（特别针对代码）
        }
        
        # 如果提供了JetBrains Mono字体路径，自动分析并调整参数
        if jetbrains_mono_path and Path(jetbrains_mono_path).exists():
            print(f"正在分析JetBrains Mono字体特征并自动调整参数...")
            self._analyze_and_adjust_parameters()
    
    def render_text_with_adjustments(self, text="The quick brown fox jumps over the lazy dog 0123456789",
                                   font_size=36, output_path="adjusted_rendering.png"):
        """
        渲染调整后的文本
        
        参数:
            text: 要渲染的文本
            font_size: 字体大小
            output_path: 输出图片路径
        """
        # 创建原始图像
        img_width = len(text) * font_size // 2 + 100
        img_height = font_size * 2 + 100
        
        # 渲染原始SF Mono
        original_img = self._render_text(text, font_size, img_width, img_height)
        
        # 渲染调整后的SF Mono
        adjusted_img = self._apply_rendering_adjustments(original_img)
        
        # 创建对比图像
        comparison_img = self._create_comparison_image(original_img, adjusted_img, text)
        
        # 保存结果
        comparison_img.save(output_path)
        print(f"调整后的渲染对比图已保存到: {output_path}")
        return comparison_img
    
    def _render_text(self, text, font_size, width, height):
        """
        使用PIL渲染文本
        """
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype(self.sf_mono_path, font_size)
            # 计算文本位置（居中）
            # 使用getbbox替代textsize（PIL 10.0+版本）
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, font=font, fill='black')
            return image
        except Exception as e:
            print(f"渲染文本时出错: {e}")
            return None
    
    def _apply_rendering_adjustments(self, image):
        """
        应用JetBrains Mono风格的渲染调整
        """
        if image is None:
            return None
        
        # 转换为OpenCV格式
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 应用二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 调整字重（使线条更粗，类似JetBrains Mono的风格）
        kernel_size = int(self.jetbrains_mono_style["weight_adjustment"] * 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 应用膨胀操作增加字重
        adjusted = cv2.dilate(binary, kernel, iterations=1)
        
        # 增强对比度
        adjusted = cv2.convertScaleAbs(adjusted, alpha=self.jetbrains_mono_style["contrast_boost"])
        
        # 锐化处理增加清晰度
        if self.jetbrains_mono_style["sharpness_factor"] > 1.0:
            blurred = cv2.GaussianBlur(adjusted, (5, 5), 1.0)
            adjusted = cv2.addWeighted(adjusted, self.jetbrains_mono_style["sharpness_factor"], 
                                      blurred, 1.0 - self.jetbrains_mono_style["sharpness_factor"], 0)
        
        # 转回RGB格式
        adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB)
        
        # 转回PIL图像
        return Image.fromarray(adjusted_rgb)
    
    def _create_comparison_image(self, original_img, adjusted_img, text):
        """
        创建原始和调整后的渲染对比图
        """
        if original_img is None or adjusted_img is None:
            return None
        
        width = max(original_img.width, adjusted_img.width)
        height = original_img.height + adjusted_img.height + 100
        
        comparison = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(comparison)
        
        # 绘制标题
        title_font = ImageFont.truetype(self.sf_mono_path, 24)
        draw.text((width//2 - 150, 20), "SF Mono (原始)", font=title_font, fill='blue')
        draw.text((width//2 - 150, original_img.height + 40), "调整后 (JetBrains Mono风格)", font=title_font, fill='red')
        
        # 粘贴图像
        comparison.paste(original_img, (0, 60))
        comparison.paste(adjusted_img, (0, original_img.height + 80))
        
        return comparison
    
    def render_code_snippet(self, code_snippet, font_size=24, output_path="code_rendering.png"):
        """
        渲染代码片段，展示调整效果
        
        参数:
            code_snippet: 要渲染的代码片段
            font_size: 字体大小
            output_path: 输出图片路径
        """
        lines = code_snippet.split('\n')
        max_line_length = max(len(line) for line in lines)
        
        img_width = max_line_length * font_size // 2 + 40
        img_height = len(lines) * font_size + 40
        
        # 渲染原始代码
        original_img = Image.new('RGB', (img_width, img_height), color='white')
        draw_original = ImageDraw.Draw(original_img)
        
        # 渲染调整后的代码（模拟调整效果）
        adjusted_img = Image.new('RGB', (img_width, img_height), color='white')
        draw_adjusted = ImageDraw.Draw(adjusted_img)
        
        try:
            font = ImageFont.truetype(self.sf_mono_path, font_size)
            
            # 绘制原始代码
            for i, line in enumerate(lines):
                draw_original.text((20, i * font_size + 20), line, font=font, fill='black')
            
            # 绘制调整后的代码（模拟调整效果）
            for i, line in enumerate(lines):
                # 模拟更粗的字重
                draw_adjusted.text((20, i * font_size + 20), line, font=font, fill='black')
                # 添加轻微的轮廓使线条看起来更粗
                draw_adjusted.text((21, i * font_size + 20), line, font=font, fill='black')
                draw_adjusted.text((20, i * font_size + 21), line, font=font, fill='black')
            
            # 应用调整效果
            adjusted_img = self._apply_rendering_adjustments(adjusted_img)
            
            # 创建对比图像
            comparison = self._create_comparison_image(original_img, adjusted_img, "代码渲染示例")
            
            # 保存结果
            comparison.save(output_path)
            print(f"代码渲染对比图已保存到: {output_path}")
            return comparison
            
        except Exception as e:
            print(f"渲染代码片段时出错: {e}")
            return None
    
    def batch_process_chars(self, chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                          font_size=48, output_path="chars_comparison.png"):
        """
        批量处理字符，展示每个字符的调整效果
        """
        chars_per_row = 10
        num_rows = (len(chars) + chars_per_row - 1) // chars_per_row
        
        char_width = font_size * 2
        char_height = font_size * 3
        
        img_width = chars_per_row * char_width
        img_height = num_rows * char_height
        
        # 创建对比图像
        comparison = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype(self.sf_mono_path, font_size)
            
            for i, char in enumerate(chars):
                row = i // chars_per_row
                col = i % chars_per_row
                
                x = col * char_width
                y = row * char_height
                
                # 绘制原始字符
                original_char_img = Image.new('RGB', (char_width, char_height), color='white')
                char_draw = ImageDraw.Draw(original_char_img)
                char_draw.text((char_width//2 - font_size//4, char_height//4), char, font=font, fill='black')
                
                # 绘制调整后的字符
                adjusted_char_img = self._apply_rendering_adjustments(original_char_img.copy())
                
                # 粘贴到对比图
                comparison.paste(original_char_img, (x, y))
                if adjusted_char_img:
                    comparison.paste(adjusted_char_img, (x, y + char_height//2))
                
                # 添加标签
                small_font = ImageFont.truetype(self.sf_mono_path, 12)
                draw.text((x + 5, y + 5), "原始", font=small_font, fill='blue')
                draw.text((x + 5, y + char_height//2 + 5), "调整后", font=small_font, fill='red')
            
            # 保存结果
            comparison.save(output_path)
            print(f"字符渲染对比图已保存到: {output_path}")
            return comparison
            
        except Exception as e:
            print(f"批量处理字符时出错: {e}")
            return None
    
    def _analyze_and_adjust_parameters(self):
        """
        分析JetBrains Mono字体的特征，并基于这些特征自动调整SF Mono的渲染参数
        """
        try:
            # 渲染对比文本
            sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            font_size = 48
            
            # 创建临时图像来分析两种字体
            width = len(sample_text) * font_size // 2
            height = font_size * 2
            
            # 渲染SF Mono
            sf_mono_img = Image.new('RGB', (width, height), color='white')
            sf_mono_draw = ImageDraw.Draw(sf_mono_img)
            sf_mono_font = ImageFont.truetype(self.sf_mono_path, font_size)
            sf_mono_draw.text((10, 10), sample_text, font=sf_mono_font, fill='black')
            
            # 渲染JetBrains Mono
            jetbrains_img = Image.new('RGB', (width, height), color='white')
            jetbrains_draw = ImageDraw.Draw(jetbrains_img)
            jetbrains_font = ImageFont.truetype(self.jetbrains_mono_path, font_size)
            jetbrains_draw.text((10, 10), sample_text, font=jetbrains_font, fill='black')
            
            # 转换为灰度图像
            sf_mono_gray = np.array(sf_mono_img.convert('L'))
            jetbrains_gray = np.array(jetbrains_img.convert('L'))
            
            # 应用二值化
            _, sf_mono_binary = cv2.threshold(sf_mono_gray, 127, 255, cv2.THRESH_BINARY_INV)
            _, jetbrains_binary = cv2.threshold(jetbrains_gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 计算字符粗细比例（通过计算黑色像素的数量）
            sf_mono_density = np.sum(sf_mono_binary == 255) / (width * height)
            jetbrains_density = np.sum(jetbrains_binary == 255) / (width * height)
            
            # 计算需要调整的比例
            if sf_mono_density > 0:
                weight_ratio = jetbrains_density / sf_mono_density
                # 限制调整范围，避免过度调整
                self.jetbrains_mono_style["weight_adjustment"] = min(max(weight_ratio, 0.8), 1.5)
            
            # 分析边缘清晰度
            # 使用Canny边缘检测
            sf_mono_edges = cv2.Canny(sf_mono_gray, 100, 200)
            jetbrains_edges = cv2.Canny(jetbrains_gray, 100, 200)
            
            # 计算边缘密度
            sf_mono_edge_density = np.sum(sf_mono_edges > 0) / (width * height)
            jetbrains_edge_density = np.sum(jetbrains_edges > 0) / (width * height)
            
            if sf_mono_edge_density > 0:
                sharpness_ratio = jetbrains_edge_density / sf_mono_edge_density
                # 调整锐度因子
                self.jetbrains_mono_style["sharpness_factor"] = min(max(sharpness_ratio, 1.0), 2.0)
            
            # 分析对比度
            sf_mono_contrast = np.max(sf_mono_gray) - np.min(sf_mono_gray)
            jetbrains_contrast = np.max(jetbrains_gray) - np.min(jetbrains_gray)
            
            if sf_mono_contrast > 0:
                contrast_ratio = jetbrains_contrast / sf_mono_contrast
                # 调整对比度因子
                self.jetbrains_mono_style["contrast_boost"] = min(max(contrast_ratio, 1.0), 1.5)
            
            print("自动调整参数完成:")
            for param, value in self.jetbrains_mono_style.items():
                print(f"  {param}: {value:.2f}")
                
        except Exception as e:
            print(f"自动分析和调整参数时出错: {e}")
            print("使用默认参数继续...")
    
    def render_comparison_with_jetbrains(self, text="The quick brown fox jumps over the lazy dog 0123456789",
                                       font_size=36, output_path="jetbrains_comparison.png"):
        """
        渲染SF Mono（原始和调整后）与JetBrains Mono的三向对比
        
        参数:
            text: 要渲染的文本
            font_size: 字体大小
            output_path: 输出图片路径
        """
        if not self.jetbrains_mono_path or not Path(self.jetbrains_mono_path).exists():
            print("错误：JetBrains Mono字体文件不存在，无法进行三向对比")
            return None
        
        # 创建原始图像
        img_width = len(text) * font_size // 2 + 100
        img_height = font_size * 2 + 50
        
        # 渲染原始SF Mono
        original_img = self._render_text(text, font_size, img_width, img_height)
        
        # 渲染调整后的SF Mono
        adjusted_img = self._apply_rendering_adjustments(original_img)
        
        # 渲染JetBrains Mono
        jetbrains_img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(jetbrains_img)
        try:
            font = ImageFont.truetype(self.jetbrains_mono_path, font_size)
            # 使用getbbox替代textsize（PIL 10.0+版本）
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img_width - text_width) // 2
            y = (img_height - text_height) // 2
            draw.text((x, y), text, font=font, fill='black')
        except Exception as e:
            print(f"渲染JetBrains Mono时出错: {e}")
            return None
        
        # 创建三向对比图像
        comparison_width = max(original_img.width, adjusted_img.width, jetbrains_img.width)
        comparison_height = original_img.height * 3 + 150
        
        comparison = Image.new('RGB', (comparison_width, comparison_height), color='white')
        draw = ImageDraw.Draw(comparison)
        
        # 绘制标题
        title_font = ImageFont.truetype(self.sf_mono_path, 24)
        draw.text((comparison_width//2 - 100, 20), "SF Mono (原始)", font=title_font, fill='blue')
        draw.text((comparison_width//2 - 150, original_img.height + 40), "SF Mono (调整后)", font=title_font, fill='green')
        draw.text((comparison_width//2 - 150, original_img.height * 2 + 60), "JetBrains Mono", font=title_font, fill='red')
        
        # 粘贴图像
        comparison.paste(original_img, (0, 60))
        comparison.paste(adjusted_img, (0, original_img.height + 80))
        comparison.paste(jetbrains_img, (0, original_img.height * 2 + 100))
        
        # 保存结果
        comparison.save(output_path)
        print(f"三向对比图已保存到: {output_path}")
        return comparison
    
    def generate_adjusted_font_file(self, output_path="adjusted_sf_mono.ttf"):
        """
        生成调整后的SF Mono字体文件
        
        参数:
            output_path: 输出字体文件路径
        
        返回:
            bool: 是否成功生成字体文件
        """
        if not font_tools_available:
            print("错误: fontTools库未安装，无法生成调整后的字体文件")
            print("请使用'pip install fonttools'来安装该库")
            return False
        
        try:
            print(f"正在生成调整后的字体文件: {output_path}")
            
            # 打开原始字体文件
            font = TTFont(self.sf_mono_path)
            
            # 获取字体中的'glyf'表，包含字形轮廓信息
            if 'glyf' not in font:
                print("错误: 字体文件不包含'glyf'表，无法修改字形")
                return False
            
            glyf_table = font['glyf']
            
            # 获取字体的度量信息
            if 'hmtx' not in font:
                print("错误: 字体文件不包含'hmtx'表，无法修改字宽")
                return False
            
            hmtx_table = font['hmtx']
            
            # 获取字形顺序映射
            if 'cmap' not in font:
                print("错误: 字体文件不包含'cmap'表，无法获取字形映射")
                return False
            
            # 获取第一个可用的cmap映射
            cmap = font['cmap'].getBestCmap()
            if not cmap:
                print("错误: 无法获取字体的字符映射表")
                return False
            
            # 修改每个字形的轮廓，使线条更粗
            weight_adjustment = self.jetbrains_mono_style["weight_adjustment"]
            spacing_adjustment = self.jetbrains_mono_style["spacing_adjustment"]
            
            # 获取所有字形名称
            glyph_order = font.getGlyphOrder()
            modified_count = 0
            
            # 遍历字形顺序中的字形
            for glyph_name in glyph_order:
                try:
                    # 获取字形对象
                    glyph = glyf_table[glyph_name]
                    
                    # 跳过复合字形和空字形
                    if glyph.numberOfContours <= 0:
                        continue
                    
                    # 获取字形ID
                    glyph_id = font.getGlyphID(glyph_name)
                    
                    # 获取水平度量信息
                    try:
                        width, lsb = hmtx_table[glyph_name]
                        # 调整字符宽度
                        new_width = int(width * spacing_adjustment)
                        hmtx_table[glyph_name] = (new_width, lsb)
                    except (KeyError, IndexError):
                        # 某些字形可能没有对应的度量信息
                        pass
                    
                    # 修改字形轮廓，使其线条变粗
                    if hasattr(glyph, 'coordinates') and glyph.coordinates:
                        # 计算缩放因子，基于字重调整
                        scale_factor = 1.0 + (weight_adjustment - 1.0) * 0.2  # 避免过度缩放
                        
                        # 缩放坐标
                        # 确保使用GlyphCoordinates对象而不是普通列表
                        original_coords = glyph.coordinates
                        new_coords = GlyphCoordinates()
                        for x, y in original_coords:
                            # 围绕原点缩放坐标
                            new_x = int(x * scale_factor)
                            new_y = int(y * scale_factor)
                            new_coords.append((new_x, new_y))
                        
                        # 更新坐标
                        glyph.coordinates = new_coords
                        modified_count += 1
                except Exception as glyph_error:
                    # 跳过单个字形的错误，继续处理其他字形
                    print(f"警告: 处理字形 {glyph_name} 时出错: {glyph_error}")
                    continue
            
            print(f"成功修改了 {modified_count} 个字形")
            
            # 更新字体名称，添加调整信息
            if 'name' in font:
                name_table = font['name']
                
                # 查找并修改字体名称
                for record in name_table.names:
                    if record.nameID == 4:  # 完整字体名称
                        try:
                            name_str = record.toUnicode()
                            # 如果已经包含调整标记，不重复添加
                            if "Adjusted" not in name_str:
                                new_name = name_str + " Adjusted"
                                record.string = new_name.encode('utf-16-be')
                        except Exception as name_error:
                            print(f"警告: 更新字体名称时出错: {name_error}")
                            continue
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 保存修改后的字体文件
            font.save(output_path)
            print(f"调整后的字体文件已成功保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"生成调整后的字体文件时出错: {type(e).__name__}: {str(e)}")
            import traceback
            print("详细错误信息:")
            traceback.print_exc()
            return False


def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    # 项目中字体文件的默认路径
    default_sf_mono = script_dir / "fonts" / "SFMonoLigaturized-Medium.ttf"
    default_jetbrains_mono = script_dir / "fonts" / "JetBrainsMono-Medium.ttf"
    
    parser = argparse.ArgumentParser(description="将SF Mono字体的渲染效果调整为与JetBrains Mono字体相似的风格")
    parser.add_argument("--sf-mono", 
                       default=str(default_sf_mono), 
                       help=f"SF Mono字体文件路径（默认: {default_sf_mono}）")
    parser.add_argument("--jetbrains-mono", 
                       default=str(default_jetbrains_mono), 
                       help=f"JetBrains Mono字体文件路径（默认: {default_jetbrains_mono}）")
    parser.add_argument("--output-dir", default="./results", help="输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查SF Mono字体文件是否存在
    if not Path(args.sf_mono).exists():
        print(f"错误：找不到SF Mono字体文件 {args.sf_mono}")
        print("请确保fonts文件夹中包含SFMono-Medium.ttf文件")
        return
    
    print(f"使用SF Mono字体文件: {args.sf_mono}")
    
    # 检查JetBrains Mono字体文件是否存在
    jetbrains_mono_path = args.jetbrains_mono if Path(args.jetbrains_mono).exists() else None
    if jetbrains_mono_path:
        print(f"使用JetBrains Mono字体文件: {jetbrains_mono_path}")
    else:
        print(f"未找到JetBrains Mono字体文件 {args.jetbrains_mono}，将使用默认参数")
    
    # 创建字体渲染调整器
    adjuster = FontRendererAdjuster(args.sf_mono, jetbrains_mono_path)
    
    # 渲染文本对比
    adjuster.render_text_with_adjustments(output_path=str(output_dir / "text_comparison.png"))
    
    # 如果有JetBrains Mono字体，生成三向对比
    if jetbrains_mono_path:
        adjuster.render_comparison_with_jetbrains(output_path=str(output_dir / "jetbrains_comparison.png"))
    
    # 生成调整后的字体文件
    adjusted_font_path = str(output_dir / "adjusted_SFMonoLigaturized-Medium.ttf")
    adjuster.generate_adjusted_font_file(output_path=adjusted_font_path)
    
    # 渲染代码片段示例
    code_example = """
# 示例Python代码
class Example:
    def __init__(self, name):
        self.name = name
        
    def hello(self):
        return f"Hello, {self.name}!"
        
# 使用示例
if __name__ == "__main__":
    ex = Example("World")
    print(ex.hello())  # 输出: Hello, World!
"""
    adjuster.render_code_snippet(code_example, output_path=str(output_dir / "code_comparison.png"))
    
    # 批量处理字符
    adjuster.batch_process_chars(output_path=str(output_dir / "chars_comparison.png"))
    
    print("\n字体渲染效果调整完成！请查看生成的对比图片。")


if __name__ == "__main__":
    main()
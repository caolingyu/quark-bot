from PIL import Image, ImageDraw, ImageFont
import textwrap
import unicodedata

def is_emoji(char):
    # 使用通用的 Unicode 范围来判断是否是 Emoji
    return unicodedata.category(char).startswith('So') or \
           0x1F600 <= ord(char) <= 0x1F64F or \
           0x1F300 <= ord(char) <= 0x1F5FF or \
           0x1F680 <= ord(char) <= 0x1F6FF or \
           0x1F700 <= ord(char) <= 0x1F77F or \
           0x1F780 <= ord(char) <= 0x1F7FF or \
           0x1F800 <= ord(char) <= 0x1F8FF or \
           0x1F900 <= ord(char) <= 0x1F9FF or \
           0x1FA00 <= ord(char) <= 0x1FA6F or \
           0x1FA70 <= ord(char) <= 0x1FAFF or \
           0x2600 <= ord(char) <= 0x26FF or \
           0x2700 <= ord(char) <= 0x27BF

def create_poster(text, output_path='poster.png'):
    # 使用支持中文和其他字符的字体路径
    font_path = '/Users/lingyu/Library/Fonts/NotoSansSC-Regular.ttf'
    # 使用支持emoji的字体路径
    emoji_font_path = '/System/Library/Fonts/Apple Color Emoji.ttc'
    
    # 设置字体大小以提升清晰度
    font_size_text = 48
    font_size_title = 72

    font_text = ImageFont.truetype(font_path, size=font_size_text)
    font_title = ImageFont.truetype(font_path, size=font_size_title)
    font_emoji = ImageFont.truetype(emoji_font_path, font_size_text, encoding='unic')

    # 设置图像的初始宽高和边距
    image_width = 1600
    margin = 100
    line_spacing = 20
    max_line_chars = 40

    # 计算文本高度
    text_height = font_text.getbbox('A')[3] - font_text.getbbox('A')[1]

    # 对文本进行换行
    wrapped_text_lines = []
    for paragraph in text.split('\n'):
        wrapped_lines = textwrap.wrap(paragraph, width=max_line_chars)
        wrapped_text_lines.extend(wrapped_lines + [''])

    # 计算图像所需高度和最大宽度
    lines = len(wrapped_text_lines)
    title = "今日币圈大事件"
    title_height = font_title.getbbox(title)[3] - font_title.getbbox(title)[1]
    image_height = margin * 2 + title_height + margin // 2 + lines * (text_height + line_spacing)

    # 计算文本的最大宽度
    text_line_widths = [font_text.getbbox(line)[2] - font_text.getbbox(line)[0] for line in wrapped_text_lines if line.strip()]
    max_text_width = max(text_line_widths) if text_line_widths else 0
    image_width = max(image_width, max_text_width + 2 * margin)

    # 创建图像
    image = Image.new('RGBA', (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 绘制标题
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((image_width - title_width) / 2, margin // 2), title, font=font_title, fill="black", embedded_color=True)

    current_h = margin + title_height + margin // 2

    # 渲染正文内容，并将 emoji 显示为彩色
    for line in wrapped_text_lines:
        if line.strip():
            current_w = margin
            for char in line:
                try:
                    char_bbox = draw.textbbox((current_w, current_h), char, font=font_text)
                    if is_emoji(char):
                        draw.text((current_w, current_h), char, font=font_emoji, fill='white', embedded_color=True)  # 填充颜色设为透明，以显示emoji的原色
                    else:
                        draw.text((current_w, current_h), char, font=font_text, fill="black")
                except Exception as e:
                    print(f"Error rendering character '{char}' : {e}")
                    draw.text((current_w, current_h), char, font=font_text, fill="black", embedded_color=True)
                current_w += char_bbox[2] - char_bbox[0]
        current_h += text_height + line_spacing

    image.save(output_path)
    print(f"海报已保存为 {output_path}")

# 测试内容
text = """
📉 美国现货比特币ETF昨日净流出2.262亿美元
总结: 据Farside Investors数据显示，6月13日，美国现货比特币ETF净流出2.262亿美元，显示出市场资金流出的趋势。
✅ Holograph团队已修复漏洞，并正与多家交易所合作封锁涉及恶意账户
总结: Holograph协议的原生代币HLG在遭遇恶意攻击者利用漏洞铸造10亿HLG后，下跌超过60%。Holograph团队已经修复了漏洞，并正与多家交易所合作，封锁涉及恶意账户。
⚖️ 美国法官批准Terraform与SEC的45亿美元和解协议
总结: 美国纽约地方法院法官Jed Rakoff已批准Terraform Labs及其创始人Do Kwon与美国证券交易委员会（SEC）达成的45亿美元和解协议，禁止其进入加密行业。
🚀 英国推进金融科技和数字资产全球领导地位
总结: 加密倡导组织Stand with Crypto (SwC) 在伦敦启动，提出了一系列建议，旨在使英国成为金融科技、数字资产和代币化的全球中心。
📉 彭博ETF分析师：或将推迟此前对现货以太坊ETF S-1于7月4日左右获批的预测
总结: 彭博分析师Eric Balchunas表示，由于SEC的公司与财务部门对提交的S-1文件的评论延迟，现货以太坊ETF S-1的批准可能会推迟。
"""

create_poster(text)
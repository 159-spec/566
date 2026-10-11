"""
桌面应用启动器 - 使用 pywebview 在原生窗口中打开量化分析页面
无需安装依赖：如果 pywebview 未安装，会回退到浏览器打开
"""
import os
import sys


def main():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    if not os.path.exists(html_path):
        print(f"找不到 index.html: {html_path}")
        input("按回车退出...")
        return

    try:
        import webview
    except ImportError:
        print("pywebview 未安装，尝试用浏览器打开...")
        import webbrowser
        webbrowser.open('file:///' + html_path.replace('\\', '/'))
        return

    # 读取 HTML 内容直接加载，避免 file:// 协议的跨域限制
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    webview.create_window(
        title='A股量化回测分析 - 优化策略v2',
        html=html_content,
        width=1200,
        height=850,
        min_size=(900, 600),
        text_select=True,
    )
    webview.start(http_server=True)


if __name__ == '__main__':
    main()

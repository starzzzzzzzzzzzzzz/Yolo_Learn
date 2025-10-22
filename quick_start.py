"""
YOLO 学习快速入门
运行这个脚本来体验 YOLOv8 的基本功能
"""

import sys


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_installation():
    """检查必要的包是否已安装"""
    print_header("步骤 1: 检查环境")
    
    required_packages = {
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'numpy': 'NumPy',
        'cv2': 'OpenCV'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下包: {', '.join(missing_packages)}")
        print("\n安装命令:")
        print("  pip install torch torchvision ultralytics opencv-python numpy")
        return False
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ CUDA 可用")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"\n⚠️ CUDA 不可用（将使用 CPU，训练会较慢）")
    except:
        pass
    
    return True


def demonstrate_yolov8():
    """演示 YOLOv8 基本功能"""
    print_header("步骤 2: YOLOv8 基础演示")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print("1. 加载预训练模型...")
        model = YOLO('yolov8n.pt')  # 会自动下载
        print("   ✅ 模型加载成功！")
        
        print("\n2. 模型信息:")
        print(f"   类别数量: {len(model.names)}")
        print(f"   类别列表: {list(model.names.values())[:10]}...（显示前10个）")
        
        print("\n3. 创建测试图像...")
        # 创建一个简单的测试图像
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print("\n4. 运行推理...")
        results = model(test_img, verbose=False)
        print("   ✅ 推理完成！")
        
        print("\n5. 模型已准备就绪，可以用于:")
        print("   - 图像检测: results = model('image.jpg')")
        print("   - 视频检测: results = model('video.mp4')")
        print("   - 摄像头检测: results = model(source=0)")
        print("   - 模型训练: model.train(data='data.yaml')")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        return False


def show_menu():
    """显示交互菜单"""
    print_header("YOLO 学习菜单")
    
    menu = """
请选择你想要学习的内容:

1. 查看 YOLOv2 核心概念（理论和代码）
2. 查看 YOLOv8 使用示例
3. 查看 YOLOv2 vs YOLOv8 对比
4. 运行 YOLOv2 代码示例
5. 查看训练指南
6. 测试 YOLOv8 (需要安装 ultralytics)
0. 退出

建议学习顺序:
  初学者: 3 → 2 → 5 → 6
  有基础: 1 → 3 → 2 → 6
  只想用: 2 → 6 → 5
    """
    
    print(menu)


def run_yolov2_examples():
    """运行 YOLOv2 示例"""
    print_header("运行 YOLOv2 核心概念示例")
    
    try:
        # 导入并运行
        import yolov2_implementation as yolov2
        
        yolov2.example_anchor_generation()
        yolov2.example_prediction_decoding()
        yolov2.example_passthrough()
        
        print("\n✅ YOLOv2 示例运行完成！")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")


def test_yolov8_live():
    """实际测试 YOLOv8"""
    print_header("YOLOv8 实际测试")
    
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        print("正在加载模型...")
        model = YOLO('yolov8n.pt')
        
        # 创建一个测试图像（带有一些形状）
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        # 画一些简单的形状
        cv2.rectangle(img, (100, 100), (300, 400), (0, 0, 255), -1)
        cv2.circle(img, (500, 200), 80, (0, 255, 0), -1)
        
        print("运行推理...")
        results = model(img, verbose=False)
        
        print("\n推理结果:")
        print(f"  检测到 {len(results[0].boxes)} 个目标")
        
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            print(f"  目标 {i+1}: {class_name} (置信度: {conf:.2f})")
        
        print("\n✅ 测试完成！")
        print("\n提示: 你可以用真实图像进行测试:")
        print("  model = YOLO('yolov8n.pt')")
        print("  results = model('your_image.jpg')")
        print("  results[0].show()  # 显示结果")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def show_file_content(filename):
    """显示文件内容"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"❌ 无法读取文件 {filename}: {e}")


def main():
    """主函数"""
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │          🎯 YOLO 学习项目 - 快速入门 🎯                │
    │                                                         │
    │              YOLOv2 & YOLOv8 学习资料                   │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """)
    
    # 检查安装
    if not check_installation():
        print("\n请先安装必要的包，然后重新运行此脚本。")
        return
    
    # 主循环
    while True:
        show_menu()
        
        try:
            choice = input("请输入选项 (0-6): ").strip()
            
            if choice == '0':
                print("\n👋 再见！祝学习愉快！")
                break
                
            elif choice == '1':
                print_header("YOLOv2 核心概念")
                print("正在打开 yolov2_implementation.py...")
                print("\n建议：在编辑器中打开这个文件查看完整代码和注释")
                input("\n按 Enter 继续...")
                
            elif choice == '2':
                print_header("YOLOv8 使用示例")
                import yolov8_example
                yolov8_example.installation_guide()
                yolov8_example.basic_usage_example()
                input("\n按 Enter 继续...")
                
            elif choice == '3':
                print_header("YOLOv2 vs YOLOv8 对比")
                import comparison_demo
                comparison_demo.architecture_comparison()
                input("\n按 Enter 继续查看更多...")
                comparison_demo.performance_comparison()
                input("\n按 Enter 继续...")
                
            elif choice == '4':
                run_yolov2_examples()
                input("\n按 Enter 继续...")
                
            elif choice == '5':
                print_header("训练指南")
                print("训练指南保存在 training_guide.md 文件中")
                print("\n建议：用 Markdown 阅读器或编辑器打开此文件")
                print("\n主要内容包括:")
                print("  • 环境准备")
                print("  • 数据集准备")
                print("  • 模型训练")
                print("  • 模型评估")
                print("  • 模型部署")
                print("  • 常见问题解答")
                input("\n按 Enter 继续...")
                
            elif choice == '6':
                test_yolov8_live()
                input("\n按 Enter 继续...")
                
            else:
                print("❌ 无效选项，请重新选择")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            input("\n按 Enter 继续...")


def quick_demo():
    """快速演示（非交互式）"""
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │          🚀 YOLO 快速演示 🚀                            │
    └─────────────────────────────────────────────────────────┘
    """)
    
    print("\n📚 学习资料已创建:")
    print("  ✅ README.md - 总览和理论知识")
    print("  ✅ yolov2_implementation.py - YOLOv2 核心实现")
    print("  ✅ yolov8_example.py - YOLOv8 使用示例")
    print("  ✅ comparison_demo.py - 详细对比分析")
    print("  ✅ training_guide.md - 完整训练指南")
    
    print("\n🎯 快速开始:")
    print("  1. 阅读 README.md 了解基础知识")
    print("  2. 运行: python yolov2_implementation.py")
    print("  3. 运行: python comparison_demo.py")
    print("  4. 安装 YOLOv8: pip install ultralytics")
    print("  5. 尝试 YOLOv8: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')('image.jpg')\"")
    
    print("\n📖 推荐学习路径:")
    print("  初学者: README.md → comparison_demo.py → yolov8_example.py")
    print("  进阶者: yolov2_implementation.py → 全部文件 → training_guide.md")
    
    print("\n💡 提示:")
    print("  • 所有代码都有详细注释")
    print("  • 可以直接运行 .py 文件查看示例")
    print("  • training_guide.md 包含完整的训练教程")


if __name__ == "__main__":
    # 如果有命令行参数 --demo，只显示快速演示
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        # 否则运行交互式菜单
        main()


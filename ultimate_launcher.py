#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io

if sys.platform == 'win32':
    import os
    os.system('chcp 65001 >nul')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time
import json
import subprocess
import webbrowser
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class UltimateChessLauncher:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.trained_model_path = None
        self.training_completed = False
        self.check_training_status()
    def check_training_status(self):
        summary_file = self.project_dir / "training_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                if summary.get('training_completed', False):
                    self.training_completed = True
                    self.trained_model_path = summary.get('final_checkpoint')
                    logger.info(f"✅ Found completed training: {self.trained_model_path}")
                else:
                    logger.info("📊 Previous training found but not completed")
            except Exception as e:
                logger.warning(f"Could not read training summary: {e}")
        checkpoint_dir = self.project_dir / "models" / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("chess_model_*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                if not self.trained_model_path:
                    self.trained_model_path = str(latest_checkpoint)
                logger.info(f"📁 Found {len(checkpoints)} checkpoint(s), latest: {latest_checkpoint.name}")
    def check_dependencies(self) -> bool:
        required_packages = ['chess', 'flask', 'flask_socketio']
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        if missing:
            print(f"Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return False
        return True
    def check_database_files(self) -> list:
        pgn_files = []
        patterns = ["*.pgn", "*.pgn.gz", "*.pgn.bz2"]
        for pattern in patterns:
            pgn_files.extend(self.project_dir.glob(pattern))
        return pgn_files
    def display_banner(self):
        banner = """
🏆 ═══════════════════════════════════════════════════════════════
🧠              ULTIMATE CHESS AI PROJECT LAUNCHER
🏆 ═══════════════════════════════════════════════════════════════
🎯 Intelligent Training  🎮 Multiple Interfaces  💾 Auto-Save
🗃️ Database Processing   📊 Progress Monitoring   ⚡ One-Click Setup
        """
        print(banner)
    def display_menu(self):
        print("\n🎛️  MAIN MENU")
        print("═" * 50)
        if self.training_completed:
            print("✅ 1. 🧠 AI Model - TRAINED AND READY!")
        else:
            print("🔄 1. 🧠 Train AI Model (Smart Pipeline)")
        print("🎮 2. 🖥️  Launch Chess GUI (Desktop)")
        print("🌐 3. 🌍 Launch Web Interface (Browser)")
        print("📊 4. 📈 Training Dashboard & Analytics")
        print("🗃️  5. 📋 Database Management")
        print("⚙️  6. 🔧 System Status & Diagnostics")
        print("❌ 7. 🚪 Exit")
        print("═" * 50)
    def train_ai_model(self):
        print("\n🧠 AI MODEL TRAINING")
        print("═" * 40)
        pgn_files = self.check_database_files()
        if not pgn_files:
            print("❌ No PGN database files found!")
            print("📥 Please add a PGN file (like Lichess database) to the project directory.")
            print("💡 You can download from: https://database.lichess.org/")
            return
        print(f"📁 Found {len(pgn_files)} database file(s):")
        for i, pgn_file in enumerate(pgn_files):
            size_mb = pgn_file.stat().st_size / (1024 * 1024)
            print(f"   {i+1}. {pgn_file.name} ({size_mb:.1f} MB)")
        if len(pgn_files) == 1:
            selected_pgn = pgn_files[0]
            print(f"\n🎯 Using: {selected_pgn.name}")
        else:
            try:
                choice = int(input(f"\nSelect database (1-{len(pgn_files)}): ")) - 1
                selected_pgn = pgn_files[choice]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
        print(f"\n🚀 Starting intelligent training pipeline...")
        print(f"📊 This will:")
        print(f"   • Process {selected_pgn.name}")
        print(f"   • Train neural network with checkpoints")
        print(f"   • Run self-play improvement")
        print(f"   • Save final trained model")
        print(f"   • Resume automatically if interrupted")
        confirm = input("\n🤔 Continue? (y/N): ").lower().strip()
        if confirm != 'y':
            print("⏹️  Training cancelled")
            return
        try:
            sys.path.append(str(self.project_dir))
            from intelligent_training_pipeline import create_training_pipeline
            print(f"\n🔥 TRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("💡 You can stop anytime with Ctrl+C - progress will be saved!")
            print("="*60)
            pipeline = create_training_pipeline(str(selected_pgn))
            pipeline.run_complete_pipeline()
            self.check_training_status()
            print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("✅ Model is now ready for use in games!")
        except KeyboardInterrupt:
            print(f"\n⏹️  Training interrupted - progress saved!")
            print(f"💾 Resume anytime by running training again")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            logger.error(f"Training error: {e}", exc_info=True)
    def launch_chess_gui(self):
        print("\n🎮 LAUNCHING CHESS GUI")
        print("═" * 30)
        try:
            gui_file = self.project_dir / "chess_game.py"
            if not gui_file.exists():
                print("❌ Chess GUI file not found!")
                return
            print("🖥️  Starting desktop chess application...")
            subprocess.Popen([sys.executable, str(gui_file)])
            print("✅ Chess GUI launched successfully!")
            print("🎯 Choose your game mode in the popup window")
        except Exception as e:
            print(f"❌ Error launching GUI: {e}")
    def launch_web_interface(self):
        print("\n🌐 LAUNCHING WEB INTERFACE")
        print("═" * 35)
        try:
            web_file = self.project_dir / "enhanced_web_interface.py"
            if not web_file.exists():
                print("❌ Web interface file not found!")
                return
            print("🌍 Starting web server...")
            print("🔗 Will open browser automatically")
            print("⏹️  Press Ctrl+C to stop server")
            env = os.environ.copy()
            pythonpath = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = f"{self.project_dir}{os.pathsep}{pythonpath}"
            process = subprocess.Popen(
                [sys.executable, str(web_file)],
                env=env,
                cwd=str(self.project_dir)
            )
            time.sleep(3)
            try:
                webbrowser.open('http://localhost:5000')
                print("✅ Web interface launched!")
                print("🌐 Browser should open automatically")
            except Exception:
                print("⚠️  Web server started, but couldn't open browser")
                print("🔗 Manually visit: http://localhost:5000")
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n⏹️  Stopping web server...")
                process.terminate()
        except Exception as e:
            print(f"❌ Error launching web interface: {e}")
    def training_dashboard(self):
        print("\n📊 TRAINING DASHBOARD")
        print("═" * 30)
        db_file = self.project_dir / "training_history.db"
        if not db_file.exists():
            print("❌ No training history found")
            print("💡 Train a model first to see analytics")
            return
        try:
            sys.path.append(str(self.project_dir))
            from training_monitor import TrainingDatabase
            db = TrainingDatabase(str(db_file))
            print("📈 Training History Available")
            print("💡 Detailed analytics in training_logs/ directory")
            try:
                import matplotlib.pyplot as plt
                print("📊 Graphical analytics available")
            except ImportError:
                print("📋 Text-based analytics only")
        except Exception as e:
            print(f"❌ Error accessing training data: {e}")
    def database_management(self):
        print("\n🗃️  DATABASE MANAGEMENT")
        print("═" * 35)
        pgn_files = self.check_database_files()
        if not pgn_files:
            print("❌ No PGN files found in project directory")
        else:
            print(f"📁 Found {len(pgn_files)} database file(s):")
            for pgn_file in pgn_files:
                size_gb = pgn_file.stat().st_size / (1024**3)
                modified = datetime.fromtimestamp(pgn_file.stat().st_mtime)
                print(f"   📄 {pgn_file.name}")
                print(f"      Size: {size_gb:.2f} GB")
                print(f"      Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
                print()
        processed_file = self.project_dir / "processed_training_data.pkl"
        if processed_file.exists():
            size_mb = processed_file.stat().st_size / (1024**2)
            print(f"✅ Processed training data: {size_mb:.1f} MB")
        else:
            print("⚠️  No processed training data found")
        print("\n💡 To add new databases:")
        print("   • Download PGN files from Lichess: https://database.lichess.org/")
        print("   • Place them in the project directory")
        print("   • Run training to process automatically")
    def system_status(self):
        print("\n⚙️  SYSTEM STATUS")
        print("═" * 25)
        print("📦 Dependencies:")
        if self.check_dependencies():
            print("   ✅ All required packages installed")
        else:
            print("   ❌ Some packages missing")
        print("\n🖥️  Hardware:")
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   ✅ GPU: {gpu_name}")
            else:
                print("   ⚠️  GPU: Not available (using CPU)")
        except ImportError:
            print("   ❌ PyTorch not available")
        print("\n📁 Project Files:")
        essential_files = [
            "neural_network.py",
            "chess_game.py",
            "enhanced_web_interface.py",
            "intelligent_training_pipeline.py",
            "smart_database_parser.py"
        ]
        for file in essential_files:
            if (self.project_dir / file).exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - MISSING!")
        print(f"\n🧠 AI Model:")
        if self.training_completed:
            print(f"   ✅ Trained and ready")
            print(f"   📁 Model: {Path(self.trained_model_path).name}")
        else:
            print(f"   ⚠️  Not trained yet")
        pgn_count = len(self.check_database_files())
        print(f"\n🗃️  Databases: {pgn_count} PGN file(s) found")
    def run(self):
        self.display_banner()
        if not self.check_dependencies():
            print("\n❌ Cannot continue without required dependencies")
            return
        while True:
            self.display_menu()
            try:
                choice = input("\n🎯 Select option (1-7): ").strip()
                if choice == '1':
                    self.train_ai_model()
                elif choice == '2':
                    self.launch_chess_gui()
                elif choice == '3':
                    self.launch_web_interface()
                elif choice == '4':
                    self.training_dashboard()
                elif choice == '5':
                    self.database_management()
                elif choice == '6':
                    self.system_status()
                elif choice == '7':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                if choice in ['1', '2', '3', '4', '5', '6']:
                    input("\n⏸️  Press Enter to continue...")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("⏸️  Press Enter to continue...")
def main():
    try:
        launcher = UltimateChessLauncher()
        launcher.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
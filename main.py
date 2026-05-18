"""
main.py — FaceGuard Attendance System entry point.

Default:
    python main.py            # launches the GUI

CLI fallback:
    python main.py --cli      # legacy text menu
"""

import argparse
import sys

from database import init_db


CLI_MENU = """
========================================
       FaceGuard Attendance (CLI)
========================================
  1. Register a new face
  2. Start live attendance
  3. View / export reports
  4. Remove a registered person
  0. Exit
========================================"""


def cli():
    init_db()
    while True:
        print(CLI_MENU)
        choice = input("  Choice: ").strip()

        if choice == "1":
            from register import register_face
            name = input("  Enter name: ").strip()
            if name:
                register_face(name)
            else:
                print("  Name cannot be empty.")

        elif choice == "2":
            from recognize import run_recognition
            run_recognition()

        elif choice == "3":
            from report import menu as report_menu
            report_menu()

        elif choice == "4":
            from register import remove_face
            name = input("  Name to remove: ").strip()
            if name:
                remove_face(name)
            else:
                print("  Name cannot be empty.")

        elif choice == "0":
            print("  Goodbye.")
            sys.exit(0)

        else:
            print("  Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="FaceGuard Attendance")
    parser.add_argument("--cli", action="store_true",
                        help="Use the text menu instead of the GUI")
    args = parser.parse_args()

    if args.cli:
        cli()
    else:
        from gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()

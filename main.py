"""main.py
Chạy giao diện phân cụm khách hàng.

Cách chạy:
    pip install -r requirements.txt
    python main.py
"""
from gui_app import CustomerClusteringApp


def main():
    app = CustomerClusteringApp()
    app.mainloop()


if __name__ == "__main__":
    main()

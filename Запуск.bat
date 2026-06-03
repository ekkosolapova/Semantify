@echo off
chcp 65001 > nul
title Запуск Модуля
echo [1/2] Активация окружения

call C:\Users\katerina\anaconda3\Scripts\activate.bat C:\Users\katerina\anaconda3

echo [2/2] Инициализация интерфейса
python e:/VKR/.vscode/Semantify.py

pause

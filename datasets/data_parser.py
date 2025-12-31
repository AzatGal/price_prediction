from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.safari.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time

# Настройка опций для Safari
safari_options = Options()

# Инициализация драйвера Safari
driver = webdriver.Safari(options=safari_options)

try:
    # Устанавливаем неявное ожидание (в секундах)
    driver.implicitly_wait(10)

    # Открываем сайт
    driver.get("https://www.cian.ru")  # Замените на нужный URL
    print("Открыта главная страница")

    # Ждем загрузки страницы
    time.sleep(2)

    # Пример 1: Клик по ссылке
    try:
        # Находим ссылку по тексту, классу, ID или CSS-селектору
        link = driver.find_element(By.LINK_TEXT, "Продажа")  # Или другой текст ссылки
        link.click()
        print("Перешли на страницу 'О нас'")
        time.sleep(2)
    except Exception as e:
        print(f"Не удалось найти/кликнуть ссылку: {e}")

    # Пример 2: Клик по кнопке
    try:
        button = driver.find_element(By.CSS_SELECTOR, "#frontend-mainpage > div > section > div > div.x174413b8--b6515a--c-filters.x174413b8--b6515a--c-filters--new > div.x174413b8--b6515a--c-filters-content > div > div.x174413b8--be9afa--wrapper > div > div.x174413b8--e402d7--buttons > span > span:nth-child(2) > a")  # Используйте нужный селектор
        button.click()
        print("Клик по кнопке выполнен")
        time.sleep(2)
    except Exception as e:
        print(f"Не удалось найти/кликнуть кнопку: {e}")

    # Пример 3: Переход назад
    # driver.back()
    print("Вернулись на предыдущую страницу")
    time.sleep(2)

    # Пример 4: Переход вперед
    driver.forward()
    print("Перешли вперед")
    time.sleep(2)

    # Пример 5: Клик по элементу меню (с использованием явного ожидания)
    try:
        menu_item = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(),'Контакты')]"))
        )
        menu_item.click()
        print("Открыли страницу контактов")
        time.sleep(2)
    except Exception as e:
        print(f"Не удалось найти элемент меню: {e}")

    # Пример 6: Прокрутка страницы
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    print("Прокрутили страницу вниз")
    time.sleep(2)

    # Пример 7: Переход на другую страницу по URL
    driver.get("https://www.example.com/another-page")
    print("Перешли на другую страницу")
    time.sleep(2)

    # Пример 8: Навигация по истории
    driver.get("https://www.google.com")
    print("Перешли на Google")
    time.sleep(2)

    driver.back()
    print("Вернулись на предыдущую страницу")
    time.sleep(2)

    # Пример 9: Клик по элементу с помощью ActionChains
    try:
        element = driver.find_element(By.ID, "some-element-id")
        actions = ActionChains(driver)
        actions.move_to_element(element).click().perform()
        print("Клик выполнен с помощью ActionChains")
    except Exception as e:
        print(f"Не удалось выполнить клик через ActionChains: {e}")

    # Пример 10: Открытие ссылки в новой вкладке
    try:
        link = driver.find_element(By.PARTIAL_LINK_TEXT, "Блог")
        # Открываем ссылку в новой вкладке (Ctrl+click или Cmd+click)
        import platform

        if platform.system() == 'Darwin':  # macOS
            from selenium.webdriver.common.keys import Keys
            import pyautogui

            actions = ActionChains(driver)
            actions.key_down(Keys.COMMAND).click(link).key_up(Keys.COMMAND).perform()
        else:
            driver.execute_script("window.open(arguments[0]);", link.get_attribute('href'))
        print("Открыли ссылку в новой вкладке")
        time.sleep(2)
    except Exception as e:
        print(f"Не удалось открыть ссылку в новой вкладке: {e}")

except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    # Закрываем браузер
    input("Нажмите Enter для закрытия браузера...")
    driver.quit()
    print("Браузер закрыт")
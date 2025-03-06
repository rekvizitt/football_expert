import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from rich.logging import RichHandler
from src.config import ConfigManager

class LogManager:
    """
    Менеджер логов (Log Manager)

    Этот класс предоставляет менеджер логов, который поддерживает вывод логов в консоль и файлы.
    Логи в файлах автоматически разделяются по дням.

    Атрибуты класса:
    - logger (logging.Logger): Экземпляр логгера для записи логов.
    - log_dir (Path | None): Путь к директории логов. Если не задан, по умолчанию None.

    Методы класса:
    - __init__: Инициализирует менеджер логов и создает экземпляр логгера.
    - setup_logging: Настраивает логгер, включая уровень логирования, вывод в консоль и путь к файлам логов.
    - ensure_log_dir_exists: Проверяет, существует ли директория логов, и создает её, если необходимо.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("fexp_logger")
        self.log_dir = None
    
    def setup_logging(self, level=logging.INFO, log_to_console=False, log_path=None):
        self.logger.handlers.clear()
        self.logger.setLevel(level)
        
        if log_to_console:
            ch = RichHandler(markup=True, rich_tracebacks=True)
            ch.setFormatter(logging.Formatter("{message}", style="{", datefmt="[%X]"))
            self.logger.addHandler(ch)
        
        if log_path:
            self.log_dir = Path(log_path)
            self.ensure_log_dir_exists(self.log_dir)
            log_file_name = "fexp.txt"
            log_file = self.log_dir.joinpath(log_file_name)
            fh = TimedRotatingFileHandler (
                log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
            )
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(fh)
        
    @staticmethod
    def ensure_log_dir_exists(log_path: Path):
        log_path.mkdir(parents=True, exist_ok=True)
        
def log_setup(log_to_console=True):
    config_manager = ConfigManager()
    logging_level = config_manager.logging_level
    
    logger = logging.getLogger("fexp_logger")
    if logger.hasHandlers():
        return logger
    
    root_dir = Path(__file__).resolve().parent.parent
    temp_log_dir = root_dir / "logs"
    temp_log_dir.mkdir(exist_ok=True)
    
    log_manager = LogManager()
    log_manager.setup_logging(
        level=logging_level, log_to_console=log_to_console, log_path=temp_log_dir
    )
    
    return logger

logger = log_setup()
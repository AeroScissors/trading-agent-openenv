from env.tasks import easy, medium, hard

TASK_MAP = {
    "easy":   easy,
    "medium": medium,
    "hard":   hard,
}

ALL_TASK_INFO = [easy.TASK_INFO, medium.TASK_INFO, hard.TASK_INFO]
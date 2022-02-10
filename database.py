from main import db


class User(db.Model):
    """
    """
    __tablename__ = 'user'
    __table_args__ = {'extend_existing': True}
    username = db.Column(db.String(80), nullable=False, primary_key=True)
    train_task_id = db.Column(db.String(80), nullable=True)


if __name__ == '__main__':
    db.create_all()

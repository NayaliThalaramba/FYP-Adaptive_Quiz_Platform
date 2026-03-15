from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)

    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)

    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    role = db.Column(db.String(20), default="student")

    quiz_attempts = db.relationship('QuizAttempt', backref='user', lazy=True)


class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    score = db.Column(db.Float, nullable=False)
    time_spent = db.Column(db.Float, nullable=False)
    engagement_boost = db.Column(db.Float, nullable=False)
    reward_type = db.Column(db.String(50), nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
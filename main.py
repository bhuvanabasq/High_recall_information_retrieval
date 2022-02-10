import time

from flask import Flask, url_for, Response
from flask import request, redirect
from flask import render_template, jsonify
import pandas as pd


import model
import rq
from rq.job import Job

from worker import conn
import json
from flask_sqlalchemy import SQLAlchemy


FLASK_APP = Flask(__name__)
tasksQueue = rq.Queue(connection=conn, default_timeout=3600)

FLASK_APP.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./ir_classifier.db'
FLASK_APP.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(FLASK_APP)
from sqlalchemy import update


@FLASK_APP.route("/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task = tasksQueue.fetch_job(task_id)

    if task:
        response_object = {
            "status": "success",
            "data": {
                "task_id": task.get_id(),
                "task_status": task.get_status(),
                "task_result": task.result,
            },
        }
    else:
        response_object = {"status": "error"}

    response = json.dumps(response_object)
    return Response(response, status=200, mimetype='application/json')


@FLASK_APP.route('/progress/<string:job_id>')
def progress(job_id):
    def get_status():

        job = Job.fetch(job_id, connection=conn)
        status = job.get_status()

        while status != 'finished':

            status = job.get_status()
            job.refresh()

            d = {'status': status}

            if 'progress' in job.meta:
                d['value'] = job.meta['progress']
            else:
                d['value'] = 0

            # IF there's a result, add this to the stream
            if job.result:
                d['result'] = job.result

            json_data = json.dumps(d)
            yield f"data:{json_data}\n\n"
            time.sleep(1)

    return Response(get_status(), mimetype='text/event-stream')


@FLASK_APP.route('/')
def landing_page():

    return render_template('index.html')


@FLASK_APP.route('/user', methods=['POST', 'GET'])
def get_or_create_user():
    from database import User
    if request.method == 'POST':
        username = request.form['username']
        user_obj = User.query.filter_by(username=username).first()
        if user_obj is None:
            user_obj = User(username=username)
            db.session.add(user_obj)
            db.session.commit()
        result = {"username": username, "message": "added successfully"}
        return jsonify(result)
    if request.method == 'GET':
        user_obj = User.query.all()
        result = {"users": [user.username for user in user_obj]}
        return jsonify(result)


@FLASK_APP.route('/username_task/<username>', methods=['GET'])
def username_task(username):
    from database import User
    user_obj = User.query.filter_by(username=username).first()
    if user_obj.train_task_id:
        task = tasksQueue.fetch_job(user_obj.train_task_id)

        if task:
            response_object = {
                "status": "success",
                "data": {
                    "task_id": task.get_id(),
                    "task_status": task.get_status(),
                    "task_result": task.result,
                },
            }
        else:
            response_object = {"status": "error"}
    else:
        response_object = {"status": "error"}
    response = json.dumps(response_object)
    return jsonify(response)


@FLASK_APP.route('/train', methods=['POST'])
def train_dataset():
    from database import User
    username = request.form['username']
    file = request.files['file']
    reset_train = request.form['reset_train']
    if reset_train == 'on':
        reset_train = True
    else:
        reset_train = False
    train_df = pd.read_csv(file)
    job = tasksQueue.enqueue(model.create_model_and_train, train_df, username, reset_train, result_ttl=-1)
    result = json.dumps(job.get_id())
    # update into db
    user_obj = db.session.query(User).filter_by(username=username).first()
    user_obj.train_task_id = job.get_id()
    db.session.commit()
    return jsonify(result)


@FLASK_APP.route('/predict', methods=['POST'])
def test_dataset():
    username = request.form['username']
    file = request.files['file']
    test_df = pd.read_csv(file)
    job = tasksQueue.enqueue(model.test_model, test_df, username)
    result = {'job_id': job.get_id()}
    return jsonify(result)


@FLASK_APP.route('/download_predict', methods=['POST'])
def download_predict():
    username = request.form['username']
    predict = pd.read_csv("predict_"+username)

    return Response(predict.to_csv(), status=200, mimetype='text/csv')


if __name__ == '__main__':
    FLASK_APP.run(port=9000)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import unittest

from app import create_app
from config import TestConfig
from models import db, Note

class NoteTest(unittest.TestCase):

    def setUp(self):
        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

        with self.app.app_context():
            #db.drop_all()
            db.create_all()

    def test_create_not(self):
        with  self.app.app_context():
            note_db = Note(title="Título", content="Contenido")
            db.session.add(note_db)
            db.session.commit()

            note = Note.query.first()

            self.assertEqual(note.title, "Título")
            self.assertEqual(note.content, "Contenido")
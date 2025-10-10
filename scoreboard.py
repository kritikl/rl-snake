from turtle import Turtle

ALIGN="center"
TYPE="Courier"
STYLE="normal"
SIZE=15

class Score(Turtle):
    def __init__(self):
        super().__init__()
        self.penup()
        self.score=0
        self.color("white")
        self.goto(0,270)
        self.write(f"Score:  {self.score}", align=ALIGN, font=(TYPE, SIZE, STYLE))
        self.hideturtle()
        
    def increase_score(self):
        self.score+=1
        self.clear()
        self.write(f"Score:  {self.score}", align=ALIGN, font=(TYPE, SIZE, STYLE))
        
    def game_over(self):
        self.goto(0,0)
        self.write(f"GAME OVER!", align=ALIGN, font=(TYPE, 24, STYLE))
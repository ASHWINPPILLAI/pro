import turtle

# Set up the turtle screen
screen = turtle.Screen()
screen.setup(width=600, height=600)
screen.title("Indian National Flag")

# Create a turtle object
flag_turtle = turtle.Turtle()
flag_turtle.speed(0)
flag_turtle.penup()
flag_turtle.hideturtle()

# Draw the saffron stripe
flag_turtle.goto(-500, 200)
flag_turtle.color("#FF9933")
flag_turtle.begin_fill()
for _ in range(2):
    flag_turtle.forward(1500) # Increased width for saffron
    flag_turtle.right(90)
    flag_turtle.forward(100)
    flag_turtle.right(90)
flag_turtle.end_fill()

# Draw the white stripe
flag_turtle.goto(-500, 100)
flag_turtle.color("white")
flag_turtle.begin_fill()
for _ in range(2):
    flag_turtle.forward(1500) # Increased width for white
    flag_turtle.right(90)
    flag_turtle.forward(100)
    flag_turtle.right(90)
flag_turtle.end_fill()

# Draw the green stripe
flag_turtle.goto(-500, 0)
flag_turtle.color("#138808")
flag_turtle.begin_fill()
for _ in range(2):
    flag_turtle.forward(1500) # Increased width for green
    flag_turtle.right(90)
    flag_turtle.forward(100)
    flag_turtle.right(90)
flag_turtle.end_fill()

# Draw the Ashoka Chakra
flag_turtle.penup()
flag_turtle.goto(0, 0)
flag_turtle.pendown()
flag_turtle.color("navy")
flag_turtle.pensize(4)

# Draw the outer circle
flag_turtle.circle(50)

# Draw the 24 spokes
for _ in range(24):
    flag_turtle.penup()
    flag_turtle.goto(0, 50)
    flag_turtle.pendown()
    flag_turtle.forward(50)
    flag_turtle.backward(50)
    flag_turtle.left(360 / 24)

# Hide the turtle
flag_turtle.hideturtle()

# Keep the window open until it's manually closed
turtle.done()

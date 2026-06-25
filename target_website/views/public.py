# Flask routes: Home, Books, Book Detail, Cart, About.  Mock book data for AI middleware traffic collection.

import random
from flask import Blueprint, render_template, request, session

public_bp = Blueprint("public", __name__)

# ============================================================
# Mock book database (15 titles across 5 categories)
# ============================================================
BOOKS_DB = [
    {
        "id": 1, "title": "Computer Systems: A Programmer's Perspective (CS:APP)",
        "author": "Randal E. Bryant & David R. O'Hallaron",
        "publisher": "Pearson", "isbn": "978-0-13-409266-9",
        "category": "Computer Science", "price": 139.00, "stock": 45,
        "color1": "#2c3e50", "color2": "#3498db",
        "description": "A comprehensive introduction to computer systems from a programmer's perspective, covering data representation, machine-level code, processor architecture, memory hierarchy, linking, exceptional control flow, virtual memory, and system-level I/O."
    },
    {
        "id": 2, "title": "Introduction to Algorithms (CLRS)",
        "author": "Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest & Clifford Stein",
        "publisher": "MIT Press", "isbn": "978-0-262-04630-5",
        "category": "Computer Science", "price": 128.00, "stock": 32,
        "color1": "#0f0c29", "color2": "#302b63",
        "description": "The classic algorithms textbook covering sorting, graph algorithms, dynamic programming, greedy algorithms, and amortized analysis. The standard reference for algorithms courses worldwide."
    },
    {
        "id": 3, "title": "One Hundred Years of Solitude",
        "author": "Gabriel Garcia Marquez",
        "publisher": "Harper Perennial", "isbn": "978-0-06-088328-7",
        "category": "Literature & Fiction", "price": 55.00, "stock": 120,
        "color1": "#3e0000", "color2": "#b71c1c",
        "description": "The masterpiece of magical realism, telling the seven-generation saga of the Buendia family in the mythical town of Macondo. A sweeping reflection of a century of change across Latin America."
    },
    {
        "id": 4, "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "publisher": "HarperCollins", "isbn": "978-0-06-112008-4",
        "category": "Literature & Fiction", "price": 28.00, "stock": 200,
        "color1": "#1b1b1b", "color2": "#636363",
        "description": "Through the eyes of Scout Finch, this Pulitzer Prize-winning novel explores racial injustice in the American South during the 1930s. A timeless story of compassion, courage, and moral awakening."
    },
    {
        "id": 5, "title": "Sapiens: A Brief History of Humankind",
        "author": "Yuval Noah Harari",
        "publisher": "Harper", "isbn": "978-0-06-231611-0",
        "category": "History & Humanities", "price": 68.00, "stock": 88,
        "color1": "#004d40", "color2": "#00897b",
        "description": "From the dawn of humankind to the 21st century — how Homo sapiens rose from an unremarkable species to dominate the planet, building civilizations, empires, and complex belief systems."
    },
    {
        "id": 6, "title": "Guns, Germs, and Steel",
        "author": "Jared Diamond",
        "publisher": "W. W. Norton", "isbn": "978-0-393-31755-8",
        "category": "History & Humanities", "price": 36.00, "stock": 65,
        "color1": "#3e2723", "color2": "#8d6e63",
        "description": "Why did some civilizations conquer others? Diamond traces the environmental and geographic factors that shaped the fates of human societies across 13,000 years of history. Pulitzer Prize winner."
    },
    {
        "id": 7, "title": "The Design of Everyday Things",
        "author": "Donald A. Norman",
        "publisher": "Basic Books", "isbn": "978-0-465-05065-9",
        "category": "Art & Design", "price": 42.00, "stock": 40,
        "color1": "#1a237e", "color2": "#5c6bc0",
        "description": "Cognitive psychologist Norman analyzes usability problems in everyday objects and lays out user-centered design principles. Essential reading for UI/UX designers and anyone who makes things for people."
    },
    {
        "id": 8, "title": "The Non-Designer's Design Book",
        "author": "Robin Williams",
        "publisher": "Peachpit Press", "isbn": "978-0-13-396615-2",
        "category": "Art & Design", "price": 59.00, "stock": 25,
        "color1": "#bf360c", "color2": "#ff7043",
        "description": "A concise guide to the four basic principles of design — proximity, alignment, repetition, and contrast — with plenty of real-world examples. Helps non-designers quickly improve their visual design skills."
    },
    {
        "id": 9, "title": "Python Crash Course",
        "author": "Eric Matthes",
        "publisher": "No Starch Press", "isbn": "978-1-7185-0270-3",
        "category": "Computer Science", "price": 89.00, "stock": 150,
        "color1": "#ffd600", "color2": "#ff6d00",
        "description": "A hands-on, project-based introduction to Python. Covers fundamentals and then dives into real projects: Django web apps, data visualization, and pygame game development. Over a million copies sold worldwide."
    },
    {
        "id": 10, "title": "Dune",
        "author": "Frank Herbert",
        "publisher": "Ace", "isbn": "978-0-441-17271-9",
        "category": "Literature & Fiction", "price": 41.00, "stock": 300,
        "color1": "#000", "color2": "#37474f",
        "description": "Set on the desert planet Arrakis, Dune is the story of Paul Atreides and the struggle for control of the most valuable substance in the universe — the spice melange. Winner of both the Hugo and Nebula awards."
    },
    {
        "id": 11, "title": "Principles of Economics: Microeconomics",
        "author": "N. Gregory Mankiw",
        "publisher": "Cengage Learning", "isbn": "978-0-357-72271-8",
        "category": "Business & Economics", "price": 79.00, "stock": 55,
        "color1": "#004d1a", "color2": "#00796b",
        "description": "Harvard professor Mankiw's introductory economics textbook. Explains supply and demand, elasticity, market efficiency, externalities, and public goods with clear language and abundant real-world cases."
    },
    {
        "id": 12, "title": "Zero to One: Notes on Startups",
        "author": "Peter Thiel with Blake Masters",
        "publisher": "Crown Business", "isbn": "978-0-8041-3929-8",
        "category": "Business & Economics", "price": 45.00, "stock": 70,
        "color1": "#0d47a1", "color2": "#42a5f5",
        "description": "PayPal co-founder Peter Thiel shares his startup philosophy: true innovation is not copying from 1 to N, but creating from 0 to 1. A must-read for entrepreneurs in Silicon Valley and beyond."
    },
    {
        "id": 13, "title": "Artificial Intelligence: A Modern Approach (AIMA)",
        "author": "Stuart Russell & Peter Norvig",
        "publisher": "Pearson", "isbn": "978-0-13-461099-3",
        "category": "Computer Science", "price": 158.00, "stock": 20,
        "color1": "#311b92", "color2": "#7c4dff",
        "description": "The definitive AI textbook — covers search, reasoning, knowledge representation, machine learning, NLP, computer vision, and robotics. The standard text in over 1,400 universities worldwide."
    },
    {
        "id": 14, "title": "The Silk Roads: A New History of the World",
        "author": "Peter Frankopan",
        "publisher": "Bloomsbury", "isbn": "978-1-4088-3997-3",
        "category": "History & Humanities", "price": 48.00, "stock": 95,
        "color1": "#5d4037", "color2": "#a1887f",
        "description": "A major reassessment of world history told through the lens of the Silk Roads — the crossroads of civilizations spanning Persia, Central Asia, and beyond for over two millennia."
    },
    {
        "id": 15, "title": "Thinking with Type",
        "author": "Ellen Lupton",
        "publisher": "Princeton Architectural Press", "isbn": "978-1-56898-969-3",
        "category": "Art & Design", "price": 68.00, "stock": 18,
        "color1": "#fff", "color2": "#e0e0e0",
        "description": "A clear and concise guide to typography — letter, text, grid — with practical examples and exercises. An essential reference for graphic designers, web designers, and anyone working with type."
    },
]

CATEGORIES = ["Computer Science", "Literature & Fiction", "History & Humanities", "Art & Design", "Business & Economics"]

# Mock shopping cart (stored in session)
CART_KEY = "cart_items"


def _get_cart():
    """Read cart data from session."""
    return session.get(CART_KEY, [])


def _save_cart(cart):
    """Write cart data to session."""
    session[CART_KEY] = cart


# ============================================================
# Route: Home
# ============================================================
@public_bp.route("/")
def index():
    # Pick 4 random books as editor's picks
    featured_ids = random.sample([b["id"] for b in BOOKS_DB], 4)
    featured_books = [b for b in BOOKS_DB if b["id"] in featured_ids]
    return render_template("index.html", featured_books=featured_books)


# ============================================================
# Route: Book List (with optional category filter)
# ============================================================
@public_bp.route("/books")
@public_bp.route("/books/<category>")
def books(category=None):
    if category and category in CATEGORIES:
        filtered = [b for b in BOOKS_DB if b["category"] == category]
    else:
        filtered = BOOKS_DB
        category = None

    return render_template(
        "books.html",
        books=filtered,
        categories=CATEGORIES,
        current_category=category,
    )


# ============================================================
# Route: Book Detail
# ============================================================
@public_bp.route("/book/<int:book_id>")
def book_detail(book_id):
    book = next((b for b in BOOKS_DB if b["id"] == book_id), None)
    if book is None:
        return render_template("base.html", content="<h2>Book not found</h2>"), 404
    return render_template("book_detail.html", book=book)


# ============================================================
# Route: Shopping Cart (includes add-to-cart logic)
# ============================================================
@public_bp.route("/cart")
def cart():
    cart = _get_cart()
    added_message = None

    # Handle "add to cart"
    add_id = request.args.get("add", type=int)
    if add_id:
        book = next((b for b in BOOKS_DB if b["id"] == add_id), None)
        if book:
            existing = next((item for item in cart if item["id"] == add_id), None)
            if existing:
                existing["quantity"] += 1
            else:
                cart.append({
                    "id": book["id"],
                    "title": book["title"],
                    "author": book["author"],
                    "price": book["price"],
                    "quantity": 1,
                })
            _save_cart(cart)
            added_message = f"'{book['title']}' has been added to your cart!"

    # Handle "checkout" (demo mode — clears cart and shows a message)
    if request.args.get("checkout") == "1" and cart:
        cart = []
        _save_cart(cart)
        added_message = "Order submitted! (This is a demo site — no real transaction takes place.)"

    return render_template("cart.html", cart_items=cart, added_message=added_message)


# ============================================================
# Route: About
# ============================================================
@public_bp.route("/about")
def about():
    return render_template("about.html")

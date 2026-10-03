-- Task B: SQL business analysis
-- Run against ecommerce_hackathon_clean.db
-- Net revenue = quantity * unit_price * (1 - discount)

-- 1. Total net revenue
SELECT ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) AS total_net_revenue
FROM orders;

-- 2. Top 10 customers by total spending
SELECT
    c.customer_name,
    c.city,
    COUNT(o.order_id) AS number_of_orders,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS total_spending
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name, c.city
ORDER BY total_spending DESC
LIMIT 10;

-- 3. Category-wise revenue, order count and quantity sold
SELECT
    p.category,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS revenue,
    COUNT(o.order_id) AS order_count,
    SUM(o.quantity) AS quantity_sold
FROM orders o
JOIN products p ON p.product_id = o.product_id
GROUP BY p.category
ORDER BY revenue DESC;

-- 4. Monthly net revenue trend
SELECT
    strftime('%Y-%m', order_date) AS month,
    ROUND(SUM(quantity * unit_price * (1.0 - discount)), 2) AS net_revenue
FROM orders
GROUP BY month
ORDER BY month;

-- 5. Top 5 products by net revenue
SELECT
    p.product_name,
    p.category,
    ROUND(SUM(o.quantity * o.unit_price * (1.0 - o.discount)), 2) AS net_revenue
FROM orders o
JOIN products p ON p.product_id = o.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY net_revenue DESC
LIMIT 5;

The dataset contains 23 columns and 10,000 entries. It includes a mix of numerical, categorical, and text data. Here are some notable columns:

- **Numerical Columns**: `average_rating`, `ratings_count`, `ratings_1`, `ratings_2`, `ratings_3`, `ratings_4`, `ratings_5`, etc.
- **Categorical/Text Columns**: `authors`, `original_title`, `language_code`, `title`, etc.
- **Columns with Missing Values**: `isbn`, `isbn13`, `original_publication_year`, `original_title`, `language_code`.

Here is the detailed statistical summary of the numerical columns in your dataset:

| Column                     | Count    | Mean        | Std Dev     | Min        | 25%         | Median      | 75%         | Max         | Range         |
|----------------------------|----------|-------------|-------------|------------|-------------|-------------|-------------|-------------|---------------|
| **book_id**                | 10,000   | 5,000.50    | 2,886.90    | 1          | 2,500.75    | 5,000.50    | 7,500.25    | 10,000      | 9,999         |
| **goodreads_book_id**       | 10,000   | 5,264,697   | 7,575,462   | 1          | 46,275.75   | 394,965.5   | 9,382,225   | 33,288,640  | 33,288,639    |
| **average_rating**          | 10,000   | 4.00        | 0.25        | 2.47       | 3.85        | 4.02        | 4.18        | 4.82        | 2.35          |
| **ratings_count**           | 10,000   | 54,001.24   | 157,370     | 2,716      | 13,568.75   | 21,155.5    | 41,053.5    | 4,780,653   | 4,777,937     |
| **ratings_1**               | 10,000   | 1,345.04    | 6,635.63    | 11         | 196         | 391         | 885         | 456,191     | 456,180       |
| **ratings_5**               | 10,000   | 23,789.81   | 79,768.89   | 754        | 5,334       | 8,836       | 17,304.5    | 3,011,543   | 3,010,789     |

This table shows metrics like **median**, **range**, and **quantiles** (25th and 75th percentiles) for each column.

### **Columns with Missing Values**
| Column Name                 | Missing Entries | Percentage (%) |
|-----------------------------|-----------------|----------------|
| `language_code`             | 1,084           | 10.84          |
| `isbn`                      | 700             | 7.00           |
| `isbn13`                    | 585             | 5.85           |
| `original_title`            | 585             | 5.85           |
| `original_publication_year` | 21              | 0.21           |

---

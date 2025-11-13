# GitHub Pages Setup Guide

This guide will help you deploy the Assignment 4 frontend to GitHub Pages.

---

## Option 1: Using ASN4 Folder as Root (Recommended)

### Step 1: Push Your Code to GitHub

```bash
# Navigate to the main repository
cd "C:\Users\natsu\OneDrive\Desktop\CS421\Natural-Language-Processing"

# Add all ASN4 files
git add ASN4/

# Commit the changes
git commit -m "Add Assignment 4: NER, TF-IDF, and PPMI with frontend demo"

# Push to GitHub
git push origin main
```

### Step 2: Enable GitHub Pages

1. Go to your GitHub repository: `https://github.com/yourusername/Natural-Language-Processing`
2. Click on **Settings** (top right)
3. Scroll down to **Pages** (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/(root)`
5. Click **Save**

### Step 3: Access Your Site

Your Assignment 4 demo will be available at:
```
https://yourusername.github.io/Natural-Language-Processing/ASN4/
```

---

## Option 2: Create a Separate gh-pages Branch for ASN4

If you want ASN4 to have its own dedicated page:

### Step 1: Create a gh-pages Branch

```bash
# Navigate to the main repository
cd "C:\Users\natsu\OneDrive\Desktop\CS421\Natural-Language-Processing"

# Create and checkout a new branch
git checkout --orphan gh-pages-asn4

# Remove all files from staging
git rm -rf .

# Copy only ASN4 files to root
cp -r ASN4/* .

# Add the files
git add .

# Commit
git commit -m "Deploy ASN4 to GitHub Pages"

# Push the branch
git push origin gh-pages-asn4
```

### Step 2: Configure GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Select:
   - Branch: `gh-pages-asn4`
   - Folder: `/(root)`
3. Click **Save**

### Step 3: Access Your Site

```
https://yourusername.github.io/Natural-Language-Processing/
```

---

## Option 3: Using GitHub Pages with Custom Path

If your main repository already has a GitHub Pages site:

### Step 1: Configure for Subdirectory

Your `index.html` is already configured to work from a subdirectory.

### Step 2: Update Links (if needed)

If the notebook and code links don't work, update the paths in `index.html`:

```html
<!-- Change from: -->
<a href="assignment4_showcase.ipynb" class="btn btn-primary">View Jupyter Notebook</a>

<!-- To: -->
<a href="https://github.com/yourusername/Natural-Language-Processing/blob/main/ASN4/assignment4_showcase.ipynb" class="btn btn-primary">View Jupyter Notebook</a>
```

### Step 3: Access Your Site

```
https://yourusername.github.io/Natural-Language-Processing/ASN4/
```

---

## Testing Locally Before Deployment

Before pushing to GitHub, test locally:

```bash
# Navigate to ASN4 folder
cd ASN4

# On Windows (using Python's HTTP server)
python -m http.server 8000

# On macOS/Linux
python3 -m http.server 8000
```

Then visit: `http://localhost:8000/index.html`

---

## Customization Before Deployment

### Update Personal Information

Edit `index.html` and `README.md` to replace:

1. **GitHub URLs:**
   ```html
   https://github.com/yourusername/Natural-Language-Processing
   ```
   Replace `yourusername` with your actual GitHub username.

2. **Author Information:**
   - Your name
   - Your LinkedIn profile
   - Your email

3. **Repository Links:**
   Update all repository links to point to your actual repo.

### Update README.md

Replace placeholders in `README.md`:
- `[Your Name]`
- `[Your University]`
- `[your-email@example.com]`
- GitHub Pages URL

---

## Viewing Jupyter Notebooks on GitHub

GitHub automatically renders `.ipynb` files! Users can:

1. Click on `assignment4_showcase.ipynb` in your repo
2. GitHub will display it with all outputs
3. For interactive version, use:
   - Google Colab: Add your notebook URL to `https://colab.research.google.com/github/`
   - nbviewer: `https://nbviewer.org/github/yourusername/Natural-Language-Processing/blob/main/ASN4/assignment4_showcase.ipynb`

---

## Adding a Custom Domain (Optional)

If you have a custom domain:

1. Go to **Settings** → **Pages**
2. Under **Custom domain**, enter your domain
3. Click **Save**
4. Update your DNS settings with your domain provider

---

## Troubleshooting

### Issue: Page shows 404

**Solution:**
- Wait 2-5 minutes after enabling GitHub Pages
- Check that `index.html` is in the correct folder
- Verify branch and folder settings in Pages configuration

### Issue: CSS not loading

**Solution:**
- CSS is embedded in the HTML file, so this shouldn't happen
- If using external CSS, check file paths are relative

### Issue: Links don't work

**Solution:**
- Use relative paths for local files
- Use absolute GitHub URLs for code files
- Example:
  ```html
  <!-- Local file (relative) -->
  <a href="index.html">Home</a>

  <!-- GitHub file (absolute) -->
  <a href="https://github.com/user/repo/blob/main/ASN4/HW4.py">Code</a>
  ```

---

## Alternative: Use GitHub's Built-in Jupyter Rendering

Instead of deploying a separate page, you can simply:

1. Push your code to GitHub
2. Share the direct link to your Jupyter notebook:
   ```
   https://github.com/yourusername/Natural-Language-Processing/blob/main/ASN4/assignment4_showcase.ipynb
   ```
3. GitHub will render it beautifully with all visualizations!

---

## Quick Commands Reference

```bash
# Add and commit changes
git add ASN4/
git commit -m "Add Assignment 4 with frontend demo"

# Push to main branch
git push origin main

# Create gh-pages branch (if needed)
git checkout -b gh-pages
git push origin gh-pages

# Check current branch
git branch

# Switch back to main
git checkout main
```

---

## Final Checklist

- [ ] Replace all `yourusername` placeholders
- [ ] Update personal information (name, email, LinkedIn)
- [ ] Test all links in `index.html`
- [ ] Verify Jupyter notebook displays correctly on GitHub
- [ ] Test the site locally before deployment
- [ ] Commit and push all changes
- [ ] Enable GitHub Pages in repository settings
- [ ] Wait 2-5 minutes for deployment
- [ ] Visit the live URL to verify

---

**Your Assignment 4 demo will be live at:**
```
https://yourusername.github.io/Natural-Language-Processing/ASN4/
```

**Congratulations! Your NLP project is now showcased professionally!**

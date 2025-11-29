# 🔐 Login Guide for Agent B

This guide explains how to handle authentication when automating tasks on websites that require login.

## Overview

Agent B uses **Playwright's Storage State** feature to save and restore login sessions. This approach:
- ✅ Lets you log in at your own pace (no rushing!)
- ✅ Works with 2FA, CAPTCHA, and complex login flows
- ✅ Saves cookies, localStorage, and sessionStorage to a JSON file
- ✅ Reuses the same login across multiple automation runs
- ✅ Works with any website

## Quick Start

### Step 1: Save Your Login Session (One-time Setup)

Use the `login.py` script to log in manually and save your session:

```bash
python login.py --url https://notion.so --output notion_state.json
```

**What happens:**
1. A browser window opens and navigates to the URL
2. You log in manually (take your time! no rush!)
3. Press ENTER in the terminal when you're done
4. Your login session is saved to `notion_state.json`

### Step 2: Use the Saved Session

Now run Agent B with the saved session:

```bash
python run_agentb.py "Create a database in Notion" --storage-state notion_state.json
```

**What happens:**
- Agent B loads the cookies and tokens from `notion_state.json`
- You're automatically logged in!
- The task executes while authenticated

## Examples

### For Notion
```bash
# One-time: Save your Notion login
python login.py --url https://notion.so --output notion_state.json

# Use it for automation
python run_agentb.py "Create a database called Projects" --storage-state notion_state.json
python run_agentb.py "Add a new row to Projects database" --storage-state notion_state.json
```

### For GitHub
```bash
# One-time: Save your GitHub login
python login.py --url https://github.com --output github_state.json

# Use it for automation
python run_agentb.py "Create a new repository called my-project" --storage-state github_state.json
python run_agentb.py "Create an issue in my-project repo" --storage-state github_state.json
```

### For Any Site
```bash
# One-time: Save your login for any website
python login.py --url https://app.example.com --output example_state.json

# Use it for automation
python run_agentb.py "Your task here" --storage-state example_state.json
```

## Session Management

### When to Re-login

You'll need to run `login.py` again if:
- Your session expires (typically after days/weeks)
- You change your password
- The site logs you out for security reasons

Simply run the login script again to refresh your session:
```bash
python login.py --url https://notion.so --output notion_state.json
```

### Multiple Accounts

You can save different sessions for different accounts:

```bash
# Work account
python login.py --url https://notion.so --output notion_work.json

# Personal account
python login.py --url https://notion.so --output notion_personal.json

# Use them separately
python run_agentb.py "Create work database" --storage-state notion_work.json
python run_agentb.py "Create personal note" --storage-state notion_personal.json
```

### Security Considerations

⚠️ **Important**: Storage state files contain sensitive authentication data!

1. **Never commit them to git** - Add to `.gitignore`:
   ```gitignore
   *_state.json
   *.state.json
   ```

2. **Keep them secure** - These files can access your accounts

3. **Don't share them** - They're like passwords for your session

## Alternative: Browser Profile

If you prefer, you can use `--user-data-dir` to save the entire browser profile:

```bash
python run_agentb.py "Create a database" --user-data-dir ./browser_profile
```

This saves everything (history, extensions, settings) - but is less portable than storage state files.

## Troubleshooting

### "Storage State: not found - will run without login"

The file path is wrong or file doesn't exist. Check:
```bash
ls -l notion_state.json  # Verify file exists
python run_agentb.py "task" --storage-state ./notion_state.json  # Use correct path
```

### "Login required" during task execution

Your session expired. Re-run the login script:
```bash
python login.py --url https://notion.so --output notion_state.json
```

### 2FA / CAPTCHA Issues

The storage state approach handles these perfectly:
1. Run `login.py`
2. Complete 2FA/CAPTCHA in the browser
3. Press ENTER to save the authenticated session
4. Future runs skip 2FA/CAPTCHA (session is already authenticated)

## Technical Details

### What Gets Saved?

The storage state JSON file contains:
- 🍪 **Cookies** - Session tokens, authentication cookies
- 💾 **localStorage** - Client-side storage data
- 💾 **sessionStorage** - Temporary session data
- 🌐 **Origins** - Which domains the data belongs to

### How It Works

1. **Playwright** captures the browser's authentication state
2. Serializes it to JSON format
3. On future runs, Playwright recreates the exact same state
4. Website thinks you never logged out!

### Playwright Documentation

For more details, see: https://playwright.dev/docs/auth#basic-shared-account-in-all-tests

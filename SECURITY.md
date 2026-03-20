# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in perceptimg, please report it responsibly.

### How to Report

**Do not** open a public GitHub issue for security vulnerabilities.

Instead, please:

1. Email the maintainer at: mriverolopez@gmail.com
2. Include `[SECURITY]` in the subject line
3. Provide a detailed description of the vulnerability

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Suggested fix (if any)

### Response Time

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity

### Disclosure Policy

- Vulnerabilities will be disclosed after a fix is released
- Credit will be given to the reporter (unless they prefer to remain anonymous)
- CVE numbers will be requested for significant vulnerabilities

## Security Best Practices

When using perceptimg:

1. **Validate input**: Always validate image files before processing
2. **Limit resources**: Set reasonable limits on image size and batch size
3. **Sandbox processing**: Consider running image processing in isolated environments
4. **Keep updated**: Use the latest version of perceptimg and its dependencies

## Dependency Security

perceptimg uses the following core dependencies:

- **Pillow**: Image processing library
- **NumPy**: Numerical computations
- **scikit-image**: Image metrics (SSIM, PSNR)

We monitor these dependencies for known vulnerabilities using:

- GitHub Dependabot
- `safety` package for Python dependencies

### Known Vulnerabilities

As of the latest release, there are no known security vulnerabilities in perceptimg or its direct dependencies.

---

Thank you for helping keep perceptimg secure!
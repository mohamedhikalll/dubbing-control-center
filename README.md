# الدبلجة

**أداة محلية مستقلة لتفريغ وترجمة ودبلجة ملفات الصوت والفيديو.**

تعمل على ملفات موجودة على جهازك: لا تحتاج إلى Gradio أو Pinokio أو روابط YouTube. معالجة المجلد وملف TXT تسلسلية؛ يكتمل الفيديو الحالي قبل بدء التالي.

الاسم الظاهر للمشروع بالعربية هو **الدبلجة**. اسم المستودع في رابط GitHub سيكون بالإنجليزية لتسهيل مشاركة الرابط فقط.

> **الحالة:** مشروع Windows محلي قيد التطوير. يلزم تثبيت الأدوات والنماذج المحلية؛ أول تشغيل قد ينزّل نماذج كبيرة. لم تُرفق النماذج أو التسجيلات الصوتية.

## الميزات

- واجهة طرفية تفاعلية: `dub.bat --open`.
- أوامر مباشرة لملف واحد، أو معالجة قائمة TXT UTF-8، أو مجلد.
- مراحل مستقلة: التقسيم، التفريغ، الترجمة، الدبلجة، أو المسار الكامل.
- استئناف آمن من الملفات المرحلية السليمة؛ الملفات الجزئية/الفاسدة يعاد إنشاؤها.
- ملف SRT ثنائي اللغة EN/AR عند تنفيذ الترجمة أو الدبلجة.
- تقسيم صوتي دون إعادة ترميز الفيديو، مع البحث المحلي عن فواصل هادئة.
- إخراج صوت مدبلج أو فيديو نهائي، مع حفظ النتائج والسجلات على القرص.

## متطلبات التشغيل

- Windows 10/11، Python 3.12، وFFmpeg/FFprobe في `PATH`.
- PyTorch المتوافق مع إصدار CUDA المستخدم وذاكرة GPU مناسبة للنموذج المختار. بطاقة 8GB VRAM تتطلب اختيار إعدادات ونماذج مناسبة؛ استهلاك الذاكرة يختلف حسب النموذج.
- مساحة قرص كافية لتنزيل النماذج ولملفات العمل. لا يحتوي هذا المستودع على الأوزان.

## تنزيل المشروع من GitHub

1. افتح صفحة المشروع: <https://github.com/mohamedhikalll/dubbing-control-center>.
2. اضغط الزر الأخضر **Code** ثم اختر **Download ZIP**.
3. بعد اكتمال التنزيل، فك ضغط الملف إلى مجلد قصير مثل `C:\Dubbing`.
4. اتبع خطوات التثبيت أدناه داخل المجلد الذي استخرجته. تنزيل ZIP يحوي كود البرنامج والوثائق فقط؛ لا يحوي Python أو FFmpeg أو نماذج الذكاء الاصطناعي أو ملفات صوت مرجعية.

## التثبيت الأولي

1. ثبّت Python 3.12 وFFmpeg/FFprobe، وتأكد أن `ffmpeg -version` و`ffprobe -version` يعملان من CMD.
2. افتح CMD داخل مجلد المشروع، ثم نفّذ `py -3.12 -m venv env` وبعده `env\Scripts\activate`.
3. راجع `requirements.txt` قبل التثبيت؛ يتضمن PyTorch CUDA 12.8 وروابط اعتماد Git متغيرة. ثبّت التبعيات على مسؤوليتك وبعد التأكد من توافق CUDA والنظام لديك: `python -m pip install -r requirements.txt`.
4. شغّل `dub.bat --check` للتحقق من البرامج والنماذج المطلوبة. نزّل النماذج المفقودة من مصادرها الرسمية قبل بدء معالجة بلا اتصال.
5. أضف فقط تسجيلًا مرجعيًا تملك حق استخدامه إلى `voices/` إذا كان إعداد الصوت يتطلب ذلك. لا ترفع التسجيلات إلى GitHub.

## الاستخدام

افتح الواجهة التفاعلية:

```bat
dub.bat --open
```

تشغيل ملف واحد:

```bat
dub.bat "D:\Videos\lecture.mp4" --task full --source-language en --target-language Arabic --target-code ar --device cuda --gender Man --output video
```

تشغيل مجلد أو ملف TXT: استخدم خيارات CLI `--batch-dir "D:\Videos"` أو `--batch-list "D:\Videos\list.txt"`. عناصر الدفعة تعمل واحدًا تلو الآخر. استخدم `dub.bat --help` لعرض الخيارات الفعلية.

تُحفظ النتائج في مجلد `<اسم المصدر>_dubbing\final` بجانب المصدر. احتفظ بمجلد مساحة العمل إذا أردت استئناف تشغيل متوقف؛ لا تحذف ملفات `status` أو المراحل المكتملة.

## الخصوصية والأمان

لا تضع مقاطع فيديو أو مخرجات أو تسجيلات صوتية أو ملفات `.env` أو مفاتيح API أو إعدادات حسابك في المستودع. تم استبعاد مجلد البيئة والنماذج والكاش والبيانات الشخصية من نسخة النشر.

## اللغة

التوثيق التفصيلي للمزايا والمخرجات متوفر في `FEATURES_AND_OUTPUTS_AR.txt`.

## License / الترخيص

لم يُحدّد ترخيص بعد. إلى أن يختار مالك المشروع ترخيصًا، لا يمنح هذا المستودع إذنًا بإعادة الاستخدام أو التعديل أو إعادة التوزيع.

---

**English**

A standalone local Windows tool for transcribing, translating, and dubbing local audio/video files. It does not require Gradio, Pinokio, or YouTube URLs. Folder and TXT batches are processed sequentially, one complete video at a time.

See the Arabic setup and usage sections above. Models, private voice recordings, environments, caches, and generated media are intentionally excluded. Review `requirements.txt` and CUDA compatibility before installation.

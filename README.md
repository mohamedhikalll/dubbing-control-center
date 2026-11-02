# الدبلجة

**أداة محلية مستقلة لتفريغ وترجمة ودبلجة ملفات الصوت والفيديو.**

تعمل على ملفات موجودة على جهازك: لا تحتاج إلى Gradio أو Pinokio أو روابط YouTube. معالجة المجلد وملف TXT تسلسلية؛ يكتمل الفيديو الحالي قبل بدء التالي.

الاسم الظاهر للمشروع بالعربية هو **الدبلجة**. اسم المستودع في رابط GitHub سيكون بالإنجليزية لتسهيل مشاركة الرابط فقط.

> **الحالة:** مشروع Windows محلي قيد التطوير. يلزم تثبيت Python وFFmpeg؛ النماذج غير مرفقة، ويقوم البرنامج بتنزيل النماذج المختارة تلقائيًا عند أول استخدام إذا لم يجدها وكان الإنترنت متاحًا. لم تُرفق أوزان النماذج أو التسجيلات الصوتية.

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
- يمكن التشغيل على المعالج CPU ولا يلزم وجود بطاقة NVIDIA؛ سيكون أبطأ، وستعتمد النماذج على ذاكرة RAM وقد تحتاج النماذج اللغوية الكبيرة إلى ذاكرة كبيرة. بطاقة NVIDIA اختيارية لتسريع المعالجة، وذاكرة VRAM المناسبة تعتمد على النموذج.
- مساحة قرص كافية لتنزيل النماذج ولملفات العمل. لا يحتوي هذا المستودع على الأوزان.

## تنزيل المشروع من GitHub

1. افتح صفحة المشروع: <https://github.com/mohamedhikalll/dubbing-control-center>.
2. اضغط الزر الأخضر **Code** ثم اختر **Download ZIP**.
3. بعد اكتمال التنزيل، فك ضغط الملف إلى مجلد قصير مثل `C:\Dubbing`.
4. اتبع خطوات التثبيت أدناه داخل المجلد الذي استخرجته. تنزيل ZIP يحوي كود البرنامج والوثائق فقط؛ لا يحوي Python أو FFmpeg أو نماذج الذكاء الاصطناعي أو ملفات صوت مرجعية. لا تحتاج عادةً إلى تنزيل أوزان النماذج يدويًا؛ التطبيق يحمّل النماذج التي تتطلبها المهمة والإعدادات المختارة عند أول استخدام.

## التثبيت الأولي

الأوامر التالية لـ **CMD** في Windows 10/11. يلزم وجود WinGet؛ اختبره بكتابة `winget --version`. إذا لم يتعرف Windows على الأمر، ثبّت **App Installer** من Microsoft Store ثم افتح CMD جديدًا. ثبّت كل برنامج فقط إذا لم يكن موجودًا بالفعل:

```bat
winget install --id Python.Python.3.12 -e --no-upgrade
winget install --id Gyan.FFmpeg -e --no-upgrade
winget install --id Git.Git -e --no-upgrade
```

Python مطلوب لتشغيل التطبيق. حزمة FFmpeg توفر `ffmpeg` و`ffprobe` اللذين يستخدمهما لمعالجة الوسائط. Git مطلوب لأن بعض المكتبات في `requirements.txt` تُثبّت مباشرة من مستودعات GitHub. الأوامر تستعمل Winget ومعرّفات الحزم الدقيقة؛ راجع [طريقة Winget الرسمية](https://learn.microsoft.com/en-us/windows/package-manager/winget/install)، و[حزمة Python 3.12](https://winstall.app/apps/Python.Python.3.12)، و[حزمة FFmpeg](https://winstall.app/apps/Gyan.FFmpeg)، و[حزمة Git](https://winstall.app/apps/Git.Git).

بعد التثبيت، أغلق CMD وافتحه من جديد، ثم تحقق من الأدوات:

```bat
py -3.12 --version
git --version
ffmpeg -version
ffprobe -version
```

إذا أردت استخدام بطاقة NVIDIA للتسريع، ثبّت تعريفها المتوافق من [صفحة NVIDIA الرسمية](https://www.nvidia.com/Download/index.aspx)، ثم تحقق من ظهوره بالأمر `nvidia-smi`. هذا اختياري: من لا يملك NVIDIA يتخطى هذه الخطوة ويختار CPU. ملف المتطلبات يثبت PyTorch المجهّز بـCUDA 12.8، لكن اختيار CPU لا يحتاج تعريف NVIDIA أو CUDA Toolkit. عند استخدام GPU لا تحتاج عادةً إلى تثبيت CUDA Toolkit كامل يدويًا؛ يكفي تعريف NVIDIA متوافق.

افتح CMD وانتقل إلى مجلد المشروع الذي فككت ضغطه؛ غيّر المسار إذا اخترت مجلدًا آخر:

```bat
cd /d C:\Dubbing
py -3.12 -m venv env
call env\Scripts\activate.bat
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

الأمر الأخير يثبت PyTorch وبقية مكتبات Python للمشروع. قد يستغرق وقتًا وينزّل ملفات كبيرة. بعد اكتماله، افحص البيئة:

```bat
dub.bat --check
```

عند بدء أول مهمة، ينزّل التطبيق تلقائيًا النماذج التي لا يجدها في الكاش (مثل Whisper والمحاذاة وLLM وTTS، بحسب المهمة والإعدادات). أبقِ الإنترنت متاحًا حتى يكتمل التنزيل؛ تحتاج الملفات الكبيرة إلى مساحة قرص واتصال مستقر، ثم تُخزّن محليًا لإعادة استخدامها. إذا لم يكن `winget` متاحًا، استخدم التنزيلات الرسمية: [Python 3.12](https://www.python.org/downloads/)، و[FFmpeg for Windows](https://www.gyan.dev/ffmpeg/builds/)، و[Git for Windows](https://git-scm.com/download/win). أضف فقط تسجيلًا مرجعيًا تملك حق استخدامه إلى `voices/` إذا تطلب إعداد الصوت ذلك، ولا ترفع التسجيلات إلى GitHub.

## الاستخدام

افتح الواجهة التفاعلية:

```bat
dub.bat --open
```

### التشغيل على المعالج CPU

لا تحتاج إلى NVIDIA كي تستخدم البرنامج. في الواجهة التفاعلية، غيّر خيار **Device** إلى **CPU** قبل بدء المهمة (الخيار 8 في قائمة الإعدادات). أو حدّد `--device cpu` في الأمر المباشر:

```bat
dub.bat "D:\Videos\lecture.mp4" --task full --device cpu --source-language en --target-language Arabic --target-code ar --gender Man --output video
```

المعالجة بالمعالج أبطأ عادةً من GPU، وقد تحتاج النماذج الكبيرة إلى RAM أكثر مما هو متاح، خصوصًا نماذج الترجمة ذات 7B أو 9B parameters؛ اختَر نموذجًا يناسب ذاكرة جهازك. لا تثبّت تعريف NVIDIA إذا كنت ستستخدم CPU فقط.

### فتح البرنامج من أي مجلد

إذا كان المشروع مفكوكًا في `C:\Dubbing`، يمكنك تشغيل الواجهة من أي مجلد بكتابة المسار الكامل في CMD:

```bat
C:\Dubbing\dub.bat --open
```

وفي PowerShell استخدم:

```powershell
& "C:\Dubbing\dub.bat" --open
```

لاستخدام الأمر المختصر `dub.bat --open` من أي مجلد، أضف مجلد المشروع الذي يحتوي `dub.bat` إلى متغير `Path` للمستخدم من إعدادات Windows، ثم افتح نافذة CMD/PowerShell جديدة. استبدل `C:\Dubbing` بالمسار الفعلي الذي اخترته عند فك الضغط.

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

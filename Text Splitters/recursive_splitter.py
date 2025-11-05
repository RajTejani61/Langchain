from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# text = """
# વિજ્ઞાનના મુખ્ય બે ક્ષેત્રો અસ્તિત્વ ધરાવે છે: (૧) નૈસર્ગિક વિજ્ઞાનો અને (૨) સામાજિક વિજ્ઞાનો. નૈસર્ગિક વિજ્ઞાનમાં ખગોળ શાસ્ત્ર, રસાયણ શાસ્ત્ર, ભૌતિક શાસ્ત્ર, ભૂસ્તરશાસ્ત્ર વગેરેનો સમાવેશ થાય છે જ્યારે સામાજિક વિજ્ઞાનોમાં સમાજશાસ્ત્ર, અર્થશાસ્ત્ર, રાજ્યશાસ્ત્ર, માનવનૃવંશશાસ્ત્ર, માનવશાસ્ત્ર વગેરેની ગણના થાય છે. ગણિત અને તર્કશાસ્ત્ર જેવાં શાસ્ત્રો વાસ્તવિક (એમ્પિરિકલ) હકીકતો ઉપર આધારિત ન હોવાથી તેમને ઔપચારિક (ફોર્મલ) વિજ્ઞાનો કહેવામાં આવે છે. તે જ રીતે ઈજનેરી વિજ્ઞાન, વૈદકશાસ્ત્ર (મૅડિકલ) અગેરેને વ્યવહારુ (એપ્લાઈડ) વિજ્ઞાનો ગણવામાં આવે છે. આમ, વિજ્ઞાનનું વિષયવસ્તુ અલગ અલગ હોવાથી વિજ્ઞાનની અનેક શાખા-પ્રશાખાઓ અસ્તિત્વમાં આવી છે, પરંતુ આ દરેક વિષયવસ્તુનો અભ્યાસ વૈજ્ઞાનિક પદ્ધતિથી જ થતો આવ્યો છે તેથી તે બધા વિજ્ઞાન તરીકે જ ઓળખાય છે
# """

splitter = RecursiveCharacterTextSplitter(
	chunk_size=100,
	chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
# print(chunks[0])
print(chunks)